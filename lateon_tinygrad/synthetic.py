from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

TERMS = (
  "retrieval", "latency", "benchmark", "tinygrad", "modernbert", "colbert", "embedding", "vector",
  "ranking", "query", "document", "kernel", "backend", "throughput", "token", "index", "search",
  "synthetic", "dataset", "batch", "memory", "precision", "inference", "encoder", "onnx", "runtime",
  "nvidia", "amd", "opencl", "metal", "cpu", "gpu", "apu", "driver",
)


def _sentence(rng: random.Random, min_words: int, max_words: int) -> str:
  count = rng.randint(min_words, max_words)
  words = [rng.choice(TERMS) for _ in range(count)]
  return " ".join(words)


def generate_records(count: int, seed: int = 1337) -> Iterator[dict[str, str | int]]:
  rng = random.Random(seed)
  for idx in range(count):
    kind = "query" if idx % 4 == 0 else "document"
    if kind == "query":
      text = _sentence(rng, 4, 12)
    else:
      text = ". ".join(_sentence(rng, 8, 24) for _ in range(rng.randint(2, 6))) + "."
    yield {"id": idx, "kind": kind, "text": text}


def write_jsonl(path: Path, count: int, seed: int = 1337) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as handle:
    for record in generate_records(count=count, seed=seed):
      handle.write(json.dumps(record, separators=(",", ":")) + "\n")
