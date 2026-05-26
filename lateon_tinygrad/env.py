from __future__ import annotations

import os
from pathlib import Path


def ensure_tinygrad_cache(cache_db: Path | None = None) -> Path:
  """Point Tinygrad's sqlite compile cache at a writable database before tinygrad is imported."""
  if cache_db is None:
    cache_db = Path(os.environ.get("CACHEDB", Path.cwd() / ".tinygrad-cache" / "cache.db"))
  cache_db = cache_db.expanduser().resolve()
  cache_db.parent.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("CACHEDB", str(cache_db))
  return cache_db
