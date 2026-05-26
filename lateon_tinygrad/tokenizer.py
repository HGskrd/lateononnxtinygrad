from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from tokenizers import Tokenizer

from .constants import DEFAULT_MODEL_DIR

TextKind = Literal["query", "document"]


@dataclass(frozen=True)
class TokenBatch:
  input_ids: np.ndarray
  attention_mask: np.ndarray
  texts: tuple[str, ...]
  kind: TextKind

  @property
  def shape(self) -> tuple[int, int]:
    return self.input_ids.shape


class LateOnTokenizer:
  def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR):
    self.model_dir = Path(model_dir)
    tokenizer_path = self.model_dir / "tokenizer.json"
    config_path = self.model_dir / "onnx_config.json"
    if not tokenizer_path.exists():
      raise FileNotFoundError(f"missing tokenizer file: {tokenizer_path}")
    if not config_path.exists():
      raise FileNotFoundError(f"missing ONNX config file: {config_path}")

    self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
    self.onnx_config = json.loads(config_path.read_text())
    self.query_prefix = self.onnx_config.get("query_prefix", "[Q] ")
    self.document_prefix = self.onnx_config.get("document_prefix", "[D] ")
    self.query_length = int(self.onnx_config.get("query_length", 32))
    self.document_length = int(self.onnx_config.get("document_length", 300))
    self.pad_token_id = int(self.onnx_config.get("pad_token_id", self.onnx_config.get("mask_token_id", 0)))
    self.pad_token = "[MASK]"

  def max_length_for(self, kind: TextKind) -> int:
    return self.query_length if kind == "query" else self.document_length

  def _prefix(self, text: str, kind: TextKind) -> str:
    return f"{self.query_prefix if kind == 'query' else self.document_prefix}{text}"

  def encode(self, texts: str | Iterable[str], kind: TextKind = "query", max_length: int | None = None) -> TokenBatch:
    if kind not in ("query", "document"):
      raise ValueError("kind must be 'query' or 'document'")
    text_tuple = (texts,) if isinstance(texts, str) else tuple(texts)
    if not text_tuple:
      raise ValueError("at least one text is required")

    length = int(max_length or self.max_length_for(kind))
    self.tokenizer.enable_truncation(max_length=length)
    self.tokenizer.enable_padding(length=length, pad_id=self.pad_token_id, pad_token=self.pad_token)
    encodings = self.tokenizer.encode_batch([self._prefix(text, kind) for text in text_tuple])

    input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
    attention_mask = np.asarray([encoding.attention_mask for encoding in encodings], dtype=np.int64)
    return TokenBatch(input_ids=input_ids, attention_mask=attention_mask, texts=text_tuple, kind=kind)
