from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .constants import CONFIG_FILES, MODEL_VARIANTS

ENV_KEYS = (
  "CACHEDB",
  "XDG_CACHE_HOME",
  "DEBUG",
  "DEBUGONNX",
  "ONNXLIMIT",
  "CACHELEVEL",
  "BEAM",
  "NOOPT",
  "DEVICE",
  "PYTHONPATH",
)


def _jsonable(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  if isinstance(value, dict):
    return {str(k): _jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(v) for v in value]
  return repr(value)


def namespace_to_dict(args: argparse.Namespace) -> dict[str, Any]:
  return {key: _jsonable(value) for key, value in vars(args).items() if not key.startswith("_")}


def package_versions() -> dict[str, str]:
  versions: dict[str, str] = {}
  for package in ("tinygrad", "numpy", "tokenizers"):
    try:
      versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      versions[package] = "not-installed"
  return versions


def tinygrad_runtime() -> dict[str, Any]:
  try:
    from tinygrad import Device

    return {
      "default": Device.DEFAULT,
      "known_devices": sorted(str(device) for device in Device._devices),
    }
  except Exception as exc:
    return {"error": f"{type(exc).__name__}: {exc}"}


def file_info(path: Path) -> dict[str, Any]:
  exists = path.exists()
  info: dict[str, Any] = {"path": str(path), "exists": exists}
  if exists:
    stat = path.stat()
    info.update({
      "size_bytes": stat.st_size,
      "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
    })
  return info


def model_files(model_dir: Path) -> dict[str, Any]:
  files = {filename: file_info(model_dir / filename) for filename in CONFIG_FILES}
  for filename in MODEL_VARIANTS.values():
    files[filename] = file_info(model_dir / filename)
  return files


def error_chain(exc: BaseException) -> list[dict[str, str]]:
  chain: list[dict[str, str]] = []
  current: BaseException | None = exc
  seen: set[int] = set()
  while current is not None and id(current) not in seen:
    seen.add(id(current))
    chain.append({"type": type(current).__name__, "message": str(current)})
    current = current.__cause__ or current.__context__
  return chain


def build_error_report(exc: BaseException, args: argparse.Namespace, phase: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
  model_dir = Path(getattr(args, "model_dir", "models/lightonai-LateOn"))
  dataset = getattr(args, "dataset", None)
  return {
    "schema": "lateon-tinygrad-error/v1",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "phase": phase,
    "command": {
      "argv": sys.argv,
      "args": namespace_to_dict(args),
    },
    "platform": {
      "python": sys.version,
      "executable": sys.executable,
      "platform": platform.platform(),
      "machine": platform.machine(),
      "processor": platform.processor(),
      "cwd": os.getcwd(),
    },
    "packages": package_versions(),
    "tinygrad": tinygrad_runtime(),
    "environment": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
    "files": {
      "dataset": file_info(Path(dataset)) if dataset is not None else None,
      "model_dir": str(model_dir),
      "model_files": model_files(model_dir),
    },
    "extra": _jsonable(extra or {}),
    "error": {
      "type": type(exc).__name__,
      "message": str(exc),
      "chain": error_chain(exc),
      "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    },
  }


def write_error_report(report: dict[str, Any], log_dir: Path) -> Path:
  log_dir.mkdir(parents=True, exist_ok=True)
  timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
  path = log_dir / f"lateon_error_{timestamp}_{os.getpid()}.json"
  path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return path
