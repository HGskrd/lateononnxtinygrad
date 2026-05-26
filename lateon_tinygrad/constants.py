from __future__ import annotations

from pathlib import Path

REPO_ID = "lightonai/LateOn"
HF_BASE_URL = "https://huggingface.co/lightonai/LateOn/resolve/main"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "lightonai-LateOn"

MODEL_VARIANTS = {
  "fp32": "model.onnx",
  "int8": "model_int8.onnx",
}

CONFIG_FILES = (
  "config.json",
  "onnx_config.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "modules.json",
  "sentence_bert_config.json",
  "config_sentence_transformers.json",
)
