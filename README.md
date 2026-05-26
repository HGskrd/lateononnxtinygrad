# LateOn Tinygrad ONNX Runner

This repository wraps the Hugging Face [`lightonai/LateOn`](https://huggingface.co/lightonai/LateOn) ONNX export with Tinygrad. LateOn is a ModernBERT-based ColBERT embedding model; the exported model takes token IDs and an attention mask and returns late-interaction token embeddings.

The implementation uses `tinygrad.nn.onnx.OnnxRunner`, so it runs through Tinygrad backends rather than ONNX Runtime or CUDA-specific code.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

The current workspace already has `tinygrad`, `tokenizers`, and `numpy` installed.

## Download Model

The full fp32 ONNX model and tokenizer/config files are expected under `models/lightonai-LateOn`.

```bash
python3 -m lateon_tinygrad.download --variant fp32
```

For smaller experiments, download the int8 export instead:

```bash
python3 -m lateon_tinygrad.download --variant int8
```

## Generate Synthetic Benchmark Data

```bash
python3 -m lateon_tinygrad.generate_synthetic --count 128 --output data/synthetic_lateon.jsonl
```

Each record is JSONL with `id`, `kind`, and `text`. Query records are prefixed with `[Q] ` during tokenization; document records are prefixed with `[D] `.

## Benchmark

Run one command that explores common hardware backends and benchmarks each one that works:

```bash
python3 -m lateon_tinygrad.probe --model-dir models/lightonai-LateOn --dataset data/synthetic_lateon.jsonl --kind query --batch-size 1 --length 32 --warmup 1 --iters 5
```

By default this asks Tinygrad for discovered usable devices, benchmarks those devices, records per-device results in `benchmarks/lateon_probe.jsonl`, writes a summary to `benchmarks/lateon_probe_summary.json`, and writes backend failure reports under `benchmarks/error_reports/`. Use `--devices CL,AMD` to force a list, `--devices all` to try common backend candidates even if discovery did not report them, or `--stop-after-first-success` when you only need the first working backend.

Run a short query benchmark on the Tinygrad default backend:

```bash
python3 -m lateon_tinygrad.benchmark --dataset data/synthetic_lateon.jsonl --kind query --batch-size 1 --length 32 --warmup 1 --iters 5
```

Run against a specific Tinygrad backend:

```bash
python3 -m lateon_tinygrad.benchmark --device NV --dataset data/synthetic_lateon.jsonl --kind query --batch-size 1 --length 32
python3 -m lateon_tinygrad.benchmark --device AMD --dataset data/synthetic_lateon.jsonl --kind query --batch-size 1 --length 32
python3 -m lateon_tinygrad.benchmark --device CL --dataset data/synthetic_lateon.jsonl --kind document --batch-size 1 --length 300
```

Useful backend notes:

- `NV` targets NVIDIA through Tinygrad's native NV backend without requiring CUDA.
- `AMD` targets Tinygrad's AMD backend when supported by the machine.
- `CL` is a portable OpenCL path that can work on some AMD APUs and integrated GPUs.
- `METAL` is useful on Apple Silicon.
- `CPU` is the safest fallback and the slowest for full-model benchmarking.

The benchmark prints JSON with latency, throughput, output shape, model path, and backend metadata. Add `--jsonl benchmarks/lateon.jsonl` to append machine-readable runs.

Tinygrad's compile cache is directed to `.tinygrad-cache/cache.db` by default, so sandboxed runs and fresh machines do not depend on a writable global cache. Use `--cache-db /path/to/cache.db` to override it.

## Field Error Reports

Benchmark failures automatically write a structured JSON report under `benchmarks/error_reports/`. The report includes the failing phase, command arguments, package versions, Tinygrad device metadata, selected Tinygrad/cache environment variables, dataset/model file sizes, and the full exception chain with traceback.

To make a report from a failing backend:

```bash
python3 -m lateon_tinygrad.benchmark --device AMD --dataset data/synthetic_lateon.jsonl --kind query --batch-size 1 --length 32 --warmup 0 --iters 1
```

The CLI prints the exact `LateOn error report: ...json` path on failure. Use `--error-log-dir /path/to/reports` to choose where those reports go, and `--print-traceback` if you also want the traceback on stderr.

## Implementation Surface

- `lateon_tinygrad.model.LateOnONNX`: Tinygrad-backed ONNX model wrapper.
- `lateon_tinygrad.tokenizer.LateOnTokenizer`: tokenizer wrapper using the model's ONNX config prefixes and lengths.
- `lateon_tinygrad.model.maxsim_matrix`: ColBERT MaxSim utility for query/document token embeddings.
- `lateon_tinygrad.generate_synthetic`: synthetic JSONL dataset generator.
- `lateon_tinygrad.benchmark`: repeatable benchmark CLI.
- `lateon_tinygrad.probe`: multi-backend benchmark probe for field hardware testing.

No Torch, Transformers, ONNX Runtime, or CUDA dependency is required for inference.
