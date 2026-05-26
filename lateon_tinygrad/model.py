from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .constants import DEFAULT_MODEL_DIR, MODEL_VARIANTS
from .env import ensure_tinygrad_cache

ensure_tinygrad_cache()


class LateOnONNX:
  """Tinygrad-backed wrapper for the LightOn LateOn ONNX export."""

  def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, variant: str = "fp32", device: str | None = None):
    if variant not in MODEL_VARIANTS:
      raise ValueError(f"unknown variant {variant!r}; choose one of {sorted(MODEL_VARIANTS)}")
    self.model_dir = Path(model_dir)
    self.variant = variant
    self.device = device
    self.model_path = self.model_dir / MODEL_VARIANTS[variant]
    if not self.model_path.exists():
      raise FileNotFoundError(
        f"missing ONNX model: {self.model_path}. Run `python3 -m lateon_tinygrad.download --variant {variant}` first."
      )

    ensure_tinygrad_cache()
    from tinygrad.nn.onnx import OnnxRunner

    self.runner = OnnxRunner(self.model_path)
    if self.device is not None:
      self.runner.to(self.device)
    self.input_names = tuple(self.runner.graph_inputs.keys())
    self.output_names = tuple(self.runner.graph_outputs)

  def _inputs(self, input_ids: np.ndarray, attention_mask: np.ndarray | None = None) -> dict[str, Any]:
    from tinygrad import Tensor

    ids = np.asarray(input_ids, dtype=np.int64)
    mask = np.ones_like(ids, dtype=np.int64) if attention_mask is None else np.asarray(attention_mask, dtype=np.int64)
    token_type_ids = np.zeros_like(ids, dtype=np.int64)

    tensors: dict[str, Any] = {}
    for name in self.input_names:
      lowered = name.lower()
      if "input_ids" in lowered or lowered == "ids":
        array = ids
      elif "attention_mask" in lowered or lowered == "mask":
        array = mask
      elif "token_type" in lowered or "type_ids" in lowered:
        array = token_type_ids
      else:
        raise ValueError(f"do not know how to feed ONNX input {name!r}; graph inputs are {self.input_names}")
      tensors[name] = Tensor(array, device=self.device)
    return tensors

  def encode_arrays(self, input_ids: np.ndarray, attention_mask: np.ndarray | None = None, realize: bool = True) -> dict[str, Any]:
    outputs = self.runner(self._inputs(input_ids, attention_mask))
    if realize:
      for tensor in outputs.values():
        if hasattr(tensor, "realize"):
          tensor.realize()
    return outputs

  def encode(self, token_batch: Any, realize: bool = True) -> dict[str, Any]:
    return self.encode_arrays(token_batch.input_ids, token_batch.attention_mask, realize=realize)


def first_output(outputs: dict[str, Any]) -> Any:
  if not outputs:
    raise ValueError("model returned no outputs")
  return next(iter(outputs.values()))


def maxsim_matrix(query_embeddings: Any, document_embeddings: Any) -> Any:
  """Compute ColBERT MaxSim scores for all query/document pairs."""
  q_batch, q_len, dim = query_embeddings.shape
  d_batch, d_len, d_dim = document_embeddings.shape
  if dim != d_dim:
    raise ValueError(f"embedding dimensions differ: {dim} vs {d_dim}")

  q = query_embeddings.reshape(q_batch, 1, q_len, 1, dim)
  d = document_embeddings.reshape(1, d_batch, 1, d_len, dim)
  return (q * d).sum(axis=-1).max(axis=3).sum(axis=2)
