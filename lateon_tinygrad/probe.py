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

DEFAULT_DEVICES = ("NV", "AMD", "CL", "METAL", "CPU")
ERROR_REPORT_RE = re.compile(r"LateOn error report:\s*(?P<path>.+)")


def _split_devices(value: str) -> list[str]:
  if value.lower() == "auto":
    return list(DEFAULT_DEVICES)
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

  result: dict[str, Any] = {
    "device": device,
    "status": "ok" if completed.returncode == 0 else "failed",
    "returncode": completed.returncode,
    "elapsed_s": elapsed,
    "command": cmd,
    "cache_db": str(cache_db),
    "error_report": report_path,
  }
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
  parser.add_argument("--devices", default="auto", help="comma-separated Tinygrad devices, or auto for NV,AMD,CL,METAL,CPU")
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

  devices = _split_devices(args.devices)
  args.cache_dir.mkdir(parents=True, exist_ok=True)
  args.error_log_dir.mkdir(parents=True, exist_ok=True)

  results: list[dict[str, Any]] = []
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
