from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import statistics
import time
from pathlib import Path
from typing import Iterable

from .constants import DEFAULT_MODEL_DIR, MODEL_VARIANTS
from .diagnostics import build_error_report, write_error_report
from .env import ensure_tinygrad_cache
from .synthetic import generate_records
from .tokenizer import LateOnTokenizer, TextKind


def _normalize_device(device: str) -> str | None:
  return None if device.lower() == "auto" else device.upper()


def _load_texts(dataset: Path | None, kind: TextKind, limit: int) -> list[str]:
  texts: list[str] = []
  if dataset is not None:
    with dataset.open("r", encoding="utf-8") as handle:
      for line in handle:
        record = json.loads(line)
        if record.get("kind") == kind:
          texts.append(str(record["text"]))
          if len(texts) >= limit:
            return texts
  if len(texts) < limit:
    for record in generate_records(limit * 4):
      if record["kind"] == kind:
        texts.append(str(record["text"]))
        if len(texts) >= limit:
          break
  return texts


def _batches(texts: list[str], batch_size: int) -> Iterable[list[str]]:
  idx = 0
  while True:
    batch = [texts[(idx + offset) % len(texts)] for offset in range(batch_size)]
    idx += batch_size
    yield batch


def _percentile(values: list[float], pct: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  rank = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
  return ordered[rank]


def _preflight_device(device: str | None) -> None:
  if device is None:
    return
  from tinygrad import Tensor

  try:
    (Tensor([1.0], device=device) + 1).realize().numpy()
  except Exception as exc:
    raise RuntimeError(f"Tinygrad device {device!r} is not usable on this machine: {type(exc).__name__}: {exc}") from exc


def _preflight_tinygrad_onnx() -> None:
  try:
    import tinygrad
  except Exception as exc:
    raise RuntimeError(f"tinygrad is not importable: {type(exc).__name__}: {exc}") from exc
  if importlib.util.find_spec("tinygrad.nn.onnx") is None:
    tinygrad_path = getattr(tinygrad, "__file__", "unknown")
    raise RuntimeError(
      "installed tinygrad does not provide tinygrad.nn.onnx.OnnxRunner; "
      f"tinygrad imported from {tinygrad_path}. Install a tinygrad build that includes tinygrad/nn/onnx.py."
    )


def _set_phase(args: argparse.Namespace, phase: str, **extra: object) -> None:
  args._diagnostic_phase = phase
  args._diagnostic_extra = {**getattr(args, "_diagnostic_extra", {}), **extra}


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Benchmark LightOn LateOn through Tinygrad's ONNX runner.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--variant", choices=sorted(MODEL_VARIANTS), default="fp32")
  parser.add_argument("--device", default="auto", help="Tinygrad device, e.g. auto, NV, AMD, CL, METAL, CPU")
  parser.add_argument("--dataset", type=Path, default=None)
  parser.add_argument("--kind", choices=("query", "document"), default="query")
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--length", type=int, default=None)
  parser.add_argument("--warmup", type=int, default=1)
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--include-tokenizer", action="store_true", help="include tokenization in measured latency")
  parser.add_argument("--jsonl", type=Path, default=None, help="append benchmark summary JSON to this file")
  parser.add_argument("--cache-db", type=Path, default=None, help="Tinygrad sqlite compile cache path")
  parser.add_argument("--error-log-dir", type=Path, default=Path("benchmarks/error_reports"), help="directory for JSON error reports")
  parser.add_argument("--print-traceback", action="store_true", help="also print full tracebacks to stderr")
  parser.add_argument("--dry-run", action="store_true", help="tokenize and validate inputs without loading the ONNX model")
  return parser


def _run(args: argparse.Namespace) -> None:
  _set_phase(args, "validate_args")
  if args.batch_size <= 0:
    raise ValueError("--batch-size must be positive")
  if args.iters <= 0:
    raise ValueError("--iters must be positive")
  if args.warmup < 0:
    raise ValueError("--warmup must be non-negative")

  _set_phase(args, "load_tokenizer")
  tokenizer = LateOnTokenizer(args.model_dir)
  max_length = int(args.length or tokenizer.max_length_for(args.kind))
  total_batches = args.warmup + args.iters
  _set_phase(args, "load_dataset", max_length=max_length, total_batches=total_batches)
  texts = _load_texts(args.dataset, args.kind, limit=max(args.batch_size * total_batches, args.batch_size))
  batch_iter = _batches(texts, args.batch_size)

  if args.dry_run:
    _set_phase(args, "tokenize_dry_run")
    sample = tokenizer.encode(next(batch_iter), kind=args.kind, max_length=max_length)
    print(json.dumps({
      "dry_run": True,
      "kind": args.kind,
      "input_shape": list(sample.input_ids.shape),
      "model_path": str(args.model_dir / MODEL_VARIANTS[args.variant]),
      "model_exists": (args.model_dir / MODEL_VARIANTS[args.variant]).exists(),
    }, indent=2))
    return

  _set_phase(args, "configure_tinygrad_cache")
  cache_db = ensure_tinygrad_cache(args.cache_db)
  _set_phase(args, "preflight_tinygrad_onnx")
  _preflight_tinygrad_onnx()
  from tinygrad import Device
  from .model import LateOnONNX, first_output

  requested_device = _normalize_device(args.device)
  _set_phase(args, "preflight_device", requested_device=requested_device or Device.DEFAULT)
  _preflight_device(requested_device)
  _set_phase(args, "load_onnx_model", requested_device=requested_device or Device.DEFAULT)
  model = LateOnONNX(args.model_dir, variant=args.variant, device=requested_device)

  _set_phase(args, "tokenize_benchmark_inputs", input_names=list(model.input_names), output_names=list(model.output_names))
  token_batches = []
  if not args.include_tokenizer:
    for _ in range(total_batches):
      token_batches.append(tokenizer.encode(next(batch_iter), kind=args.kind, max_length=max_length))

  latencies: list[float] = []
  output_shape: list[int] | None = None
  token_count = args.batch_size * max_length

  for idx in range(total_batches):
    _set_phase(args, "run_inference", iteration=idx, measured=idx >= args.warmup)
    if args.include_tokenizer:
      batch_texts = next(batch_iter)
      start = time.perf_counter()
      token_batch = tokenizer.encode(batch_texts, kind=args.kind, max_length=max_length)
      outputs = model.encode(token_batch, realize=True)
      elapsed = time.perf_counter() - start
    else:
      start = time.perf_counter()
      outputs = model.encode(token_batches[idx], realize=True)
      elapsed = time.perf_counter() - start

    if idx >= args.warmup:
      latencies.append(elapsed)
      output = first_output(outputs)
      output_shape = [int(dim) for dim in output.shape]

  _set_phase(args, "write_summary")
  summary = {
    "model": "lightonai/LateOn",
    "variant": args.variant,
    "model_path": str(model.model_path),
    "device_requested": args.device,
    "device_effective": requested_device or Device.DEFAULT,
    "cache_db": os.environ.get("CACHEDB", str(cache_db)),
    "kind": args.kind,
    "batch_size": args.batch_size,
    "sequence_length": max_length,
    "warmup": args.warmup,
    "iters": args.iters,
    "include_tokenizer": args.include_tokenizer,
    "latency_ms_mean": statistics.mean(latencies) * 1000.0,
    "latency_ms_median": statistics.median(latencies) * 1000.0,
    "latency_ms_p95": _percentile(latencies, 95) * 1000.0,
    "tokens_per_second": (token_count * len(latencies)) / sum(latencies),
    "output_names": list(model.output_names),
    "output_shape": output_shape,
  }
  print(json.dumps(summary, indent=2))

  if args.jsonl is not None:
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(summary, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> None:
  parser = _build_parser()
  args = parser.parse_args(argv)
  try:
    _run(args)
  except Exception as exc:
    phase = getattr(args, "_diagnostic_phase", "unknown")
    extra = getattr(args, "_diagnostic_extra", {})
    report = build_error_report(exc, args, phase=phase, extra=extra)
    print(f"ERROR during {phase}: {type(exc).__name__}: {exc}", file=sys.stderr)
    try:
      path = write_error_report(report, args.error_log_dir)
      print(f"LateOn error report: {path.resolve()}", file=sys.stderr)
    except Exception as report_exc:
      print(f"Failed to write LateOn error report: {type(report_exc).__name__}: {report_exc}", file=sys.stderr)
    if args.print_traceback:
      import traceback

      traceback.print_exception(type(exc), exc, exc.__traceback__)
    raise SystemExit(1) from exc


if __name__ == "__main__":
  main()
