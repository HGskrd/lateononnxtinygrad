from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from .constants import DEFAULT_MODEL_DIR, MODEL_VARIANTS

ALL_CANDIDATE_DEVICES = ("NV", "AMD", "CL", "METAL", "CUDA", "QCOM", "WEBGPU", "CPU")
ERROR_REPORT_RE = re.compile(r"LateOn error report:\s*(?P<path>.+)")


def _split_devices(value: str, discovered: list[str]) -> list[str]:
  lowered = value.lower()
  if lowered == "auto":
    return discovered
  if lowered == "all":
    return list(ALL_CANDIDATE_DEVICES)
  devices = [device.strip().upper() for device in value.split(",") if device.strip()]
  if not devices:
    raise ValueError("--devices must contain at least one Tinygrad device")
  return devices


def _json_from_stdout(stdout: str) -> dict[str, Any] | None:
  text = stdout.strip()
  if not text:
    return None
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    return None


def _error_report_path(stderr: str) -> str | None:
  for line in stderr.splitlines():
    match = ERROR_REPORT_RE.search(line)
    if match:
      return match.group("path").strip()
  return None


def _error_summary(report_path: str | None) -> dict[str, Any] | None:
  if report_path is None:
    return None
  try:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
  except Exception as exc:
    return {"report_path": report_path, "read_error": f"{type(exc).__name__}: {exc}"}
  error = report.get("error", {})
  return {
    "report_path": report_path,
    "phase": report.get("phase"),
    "type": error.get("type"),
    "message": error.get("message"),
    "chain": error.get("chain"),
  }


def _tail(text: str, lines: int = 30) -> str:
  parts = text.strip().splitlines()
  return "\n".join(parts[-lines:])


def _benchmark_command(args: argparse.Namespace, device: str, cache_db: Path) -> list[str]:
  cmd = [
    sys.executable,
    "-m",
    "lateon_tinygrad.benchmark",
    "--model-dir",
    str(args.model_dir),
    "--variant",
    args.variant,
    "--device",
    device,
    "--kind",
    args.kind,
    "--batch-size",
    str(args.batch_size),
    "--length",
    str(args.length),
    "--warmup",
    str(args.warmup),
    "--iters",
    str(args.iters),
    "--cache-db",
    str(cache_db),
    "--error-log-dir",
    str(args.error_log_dir),
  ]
  if args.dataset is not None:
    cmd += ["--dataset", str(args.dataset)]
  if args.include_tokenizer:
    cmd.append("--include-tokenizer")
  if args.print_traceback:
    cmd.append("--print-traceback")
  return cmd


def _discover_devices(args: argparse.Namespace) -> dict[str, Any]:
  cache_db = args.cache_dir / "discover.db"
  code = (
    "import json\n"
    "from pathlib import Path\n"
    "from lateon_tinygrad.env import ensure_tinygrad_cache\n"
    f"ensure_tinygrad_cache(Path({str(cache_db)!r}))\n"
    "from tinygrad import Device\n"
    "devices = list(Device.get_available_devices())\n"
    "print(json.dumps({'default': Device.DEFAULT, 'available': devices, 'known': sorted(Device._devices)}))\n"
  )
  completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False)
  if completed.returncode != 0:
    return {
      "status": "failed",
      "available": [],
      "returncode": completed.returncode,
      "stdout_tail": _tail(completed.stdout),
      "stderr_tail": _tail(completed.stderr),
    }
  try:
    data = json.loads(completed.stdout)
  except json.JSONDecodeError as exc:
    return {
      "status": "failed",
      "available": [],
      "returncode": completed.returncode,
      "stdout_tail": _tail(completed.stdout),
      "stderr_tail": f"could not parse discovery JSON: {exc}\n{_tail(completed.stderr)}",
    }
  return {"status": "ok", **data}


def _runtime_check(args: argparse.Namespace) -> dict[str, Any]:
  cache_db = args.cache_dir / "runtime_check.db"
  code = (
    "import importlib.metadata, importlib.util, json, os, shutil\n"
    "from pathlib import Path\n"
    "from lateon_tinygrad.env import ensure_tinygrad_cache\n"
    f"ensure_tinygrad_cache(Path({str(cache_db)!r}))\n"
    "import tinygrad\n"
    "spec = importlib.util.find_spec('tinygrad.nn.onnx')\n"
    "try:\n"
    "  version = importlib.metadata.version('tinygrad')\n"
    "except importlib.metadata.PackageNotFoundError:\n"
    "  version = 'unknown'\n"
    "print(json.dumps({\n"
    "  'tinygrad_file': getattr(tinygrad, '__file__', None),\n"
    "  'tinygrad_version': version,\n"
    "  'onnx_runner_available': spec is not None,\n"
    "  'onnx_runner_origin': None if spec is None else spec.origin,\n"
    "  'tools': {name: shutil.which(name) for name in ['clang', 'clang++', 'nvcc', 'ptxas', 'nvdisasm', 'clinfo']},\n"
    "  'env': {key: os.environ[key] for key in ['CUDA_PATH', 'CC'] if key in os.environ},\n"
    "}))\n"
  )
  completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False)
  if completed.returncode != 0:
    return {
      "status": "failed",
      "returncode": completed.returncode,
      "stdout_tail": _tail(completed.stdout),
      "stderr_tail": _tail(completed.stderr),
    }
  try:
    data = json.loads(completed.stdout)
  except json.JSONDecodeError as exc:
    return {
      "status": "failed",
      "returncode": completed.returncode,
      "stdout_tail": _tail(completed.stdout),
      "stderr_tail": f"could not parse runtime-check JSON: {exc}\n{_tail(completed.stderr)}",
    }
  if not data.get("onnx_runner_available"):
    data["status"] = "failed"
    data["message"] = "installed tinygrad does not provide tinygrad.nn.onnx.OnnxRunner"
  else:
    data["status"] = "ok"
  return data


def _run_device(args: argparse.Namespace, device: str) -> dict[str, Any]:
  cache_db = args.cache_dir / f"{device.lower()}.db"
  cmd = _benchmark_command(args, device=device, cache_db=cache_db)
  env = {**os.environ, "PYTHONUNBUFFERED": "1"}
  start = time.perf_counter()
  print(f"[lateon-probe] testing {device}: {' '.join(cmd)}", file=sys.stderr, flush=True)
  completed = subprocess.run(cmd, text=True, capture_output=True, env=env, check=False)
  elapsed = time.perf_counter() - start
  summary = _json_from_stdout(completed.stdout)
  report_path = _error_report_path(completed.stderr)
  error = _error_summary(report_path)

  result: dict[str, Any] = {
    "device": device,
    "status": "ok" if completed.returncode == 0 else "failed",
    "returncode": completed.returncode,
    "elapsed_s": elapsed,
    "command": cmd,
    "cache_db": str(cache_db),
    "error_report": report_path,
  }
  if error is not None:
    result["error"] = error
  if summary is not None:
    result["summary"] = summary
  if completed.returncode != 0:
    result["stderr_tail"] = _tail(completed.stderr)
    if completed.stdout.strip():
      result["stdout_tail"] = _tail(completed.stdout)
  return result


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as handle:
    for record in records:
      handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description="Probe Tinygrad backends and benchmark LateOn on each usable device.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--variant", choices=sorted(MODEL_VARIANTS), default="fp32")
  parser.add_argument("--devices", default="auto", help="auto uses Tinygrad-discovered usable devices; all tries common candidates; or pass comma-separated devices")
  parser.add_argument("--dataset", type=Path, default=None)
  parser.add_argument("--kind", choices=("query", "document"), default="query")
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--length", type=int, default=32)
  parser.add_argument("--warmup", type=int, default=1)
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--include-tokenizer", action="store_true")
  parser.add_argument("--cache-dir", type=Path, default=Path(".tinygrad-cache/probe"))
  parser.add_argument("--error-log-dir", type=Path, default=Path("benchmarks/error_reports"))
  parser.add_argument("--jsonl", type=Path, default=Path("benchmarks/lateon_probe.jsonl"), help="append one probe result per device")
  parser.add_argument("--summary-json", type=Path, default=Path("benchmarks/lateon_probe_summary.json"))
  parser.add_argument("--stop-after-first-success", action="store_true")
  parser.add_argument("--print-traceback", action="store_true")
  args = parser.parse_args(argv)

  args.cache_dir.mkdir(parents=True, exist_ok=True)
  args.error_log_dir.mkdir(parents=True, exist_ok=True)
  discovery = _discover_devices(args)
  runtime_check = _runtime_check(args)
  devices = _split_devices(args.devices, discovered=[str(device).upper() for device in discovery.get("available", [])])
  if not devices:
    print("[lateon-probe] no devices discovered; use --devices all or --devices CPU to force candidates", file=sys.stderr)

  results: list[dict[str, Any]] = []
  if runtime_check.get("status") != "ok":
    print(f"[lateon-probe] dependency check failed: {runtime_check.get('message', runtime_check)}", file=sys.stderr)
  else:
    for device in devices:
      result = _run_device(args, device)
      results.append(result)
      if args.jsonl is not None:
        _write_jsonl(args.jsonl, [result])
      if args.stop_after_first_success and result["status"] == "ok":
        break

  ok_results = [result for result in results if result["status"] == "ok"]
  failed_results = [result for result in results if result["status"] != "ok"]
  best = max(ok_results, key=lambda result: result["summary"]["tokens_per_second"]) if ok_results else None
  report = {
    "schema": "lateon-tinygrad-probe/v1",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model_dir": str(args.model_dir),
    "variant": args.variant,
    "kind": args.kind,
    "batch_size": args.batch_size,
    "sequence_length": args.length,
    "warmup": args.warmup,
    "iters": args.iters,
    "device_discovery": discovery,
    "runtime_check": runtime_check,
    "devices_requested": devices,
    "ok_count": len(ok_results),
    "failed_count": len(failed_results),
    "best_device": best["device"] if best else None,
    "best_tokens_per_second": best["summary"]["tokens_per_second"] if best else None,
    "results": results,
  }

  if args.summary_json is not None:
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  print(json.dumps(report, indent=2))
  if not ok_results:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
