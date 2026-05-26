from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import Request, urlopen

from .constants import CONFIG_FILES, DEFAULT_MODEL_DIR, HF_BASE_URL, MODEL_VARIANTS


def _download_file(url: str, destination: Path, force: bool = False) -> None:
  if destination.exists() and destination.stat().st_size > 0 and not force:
    print(f"exists {destination}")
    return

  destination.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = destination.with_suffix(destination.suffix + ".tmp")
  request = Request(url, headers={"User-Agent": "lateon-tinygrad/0.1"})

  with urlopen(request) as response, tmp_path.open("wb") as output:
    total = int(response.headers.get("Content-Length") or 0)
    received = 0
    while True:
      chunk = response.read(1024 * 1024)
      if not chunk:
        break
      output.write(chunk)
      received += len(chunk)
      if total:
        pct = received * 100.0 / total
        print(f"\r{destination.name}: {pct:5.1f}% ({received / 1024 / 1024:.1f} MiB)", end="", flush=True)
  if total:
    print()
  os.replace(tmp_path, destination)
  print(f"downloaded {destination}")


def download_lateon(model_dir: Path = DEFAULT_MODEL_DIR, variant: str = "fp32", force: bool = False) -> list[Path]:
  if variant not in MODEL_VARIANTS:
    raise ValueError(f"unknown variant {variant!r}; choose one of {sorted(MODEL_VARIANTS)}")

  files = [*CONFIG_FILES, MODEL_VARIANTS[variant]]
  downloaded: list[Path] = []
  for filename in files:
    destination = model_dir / filename
    _download_file(f"{HF_BASE_URL}/{filename}", destination, force=force)
    downloaded.append(destination)
  return downloaded


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description="Download LightOn LateOn ONNX files from Hugging Face.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--variant", choices=sorted(MODEL_VARIANTS), default="fp32")
  parser.add_argument("--force", action="store_true", help="redownload files that already exist")
  args = parser.parse_args(argv)

  download_lateon(model_dir=args.model_dir, variant=args.variant, force=args.force)


if __name__ == "__main__":
  main()
