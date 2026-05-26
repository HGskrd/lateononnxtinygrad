from __future__ import annotations

import argparse
from pathlib import Path

from .synthetic import write_jsonl


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description="Generate a deterministic synthetic LateOn benchmark dataset.")
  parser.add_argument("--count", type=int, default=128)
  parser.add_argument("--seed", type=int, default=1337)
  parser.add_argument("--output", type=Path, default=Path("data/synthetic_lateon.jsonl"))
  args = parser.parse_args(argv)

  if args.count <= 0:
    raise ValueError("--count must be positive")
  write_jsonl(args.output, count=args.count, seed=args.seed)
  print(f"wrote {args.count} records to {args.output}")


if __name__ == "__main__":
  main()
