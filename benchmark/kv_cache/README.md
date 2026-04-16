# TurboQuant KV Cache Compression for SGLang

This project integrates [TurboQuant](https://arxiv.org/abs/2502.12139) (ICLR 2026) into SGLang, enabling 2–4 bit KV cache compression with minimal accuracy loss. TurboQuant uses random orthogonal rotation + scalar quantization to achieve high compression ratios while preserving model quality.

## What Was Modified

### Core Implementation

| File | Description |
|------|-------------|
| `python/sglang/srt/layers/quantization/turboquant_kernels.py` | Core TurboQuant kernels: HadamardTransform (random orthogonal Q matrix), bit-packing/unpacking, quantize/dequantize, fused Triton dequant+scatter kernels |
| `python/sglang/srt/layers/quantization/turboquant.py` | SGLang quantization config integration |
| `python/sglang/srt/mem_cache/turboquant_memory_pool.py` | Compressed KV memory pool: selective dequantization, fused dequant path, workspace management |

### Bug Fixes (existing files)

| File | Change |
|------|--------|
| `python/sglang/srt/layers/attention/flashattention_backend.py` | Added `_set_active_kv_indices()` — FA3 backend never called it, causing workspace to stay all zeros (0.015% accuracy → 0.885%) |
| `python/sglang/srt/layers/attention/triton_backend.py` | Added `_set_active_kv_indices()` — same fix for Triton backend |
| `python/sglang/srt/layers/quantization/turboquant_kernels.py` | Fixed fused dequant path: replaced old Walsh-Hadamard inverse with `hadamard.inverse()` (Q^T) for consistency with forward Q rotation (0% → 88.5% accuracy) |
| `python/sglang/srt/layers/quantization/turboquant_kernels.py` | Fixed 1/2/3-bit general dequant: moved `hadamard.inverse()` before QJL block |
| `python/sglang/srt/layers/quantization/turboquant_kernels.py` | Added input padding for non-power-of-2 head_dim in `turboquant_quantize` |

## Quick Start

```bash
# Launch with 4-bit TurboQuant KV cache (FA3 backend, default)
CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server \
    --model-path /path/to/Qwen3-8B/ \
    --kv-cache-dtype turboquant \
    --turboquant-bits 4

# 3-bit mode
CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server \
    --model-path /path/to/Qwen3-8B/ \
    --kv-cache-dtype turboquant \
    --turboquant-bits 3

# Prod mode (3-bit MSE + 1-bit QJL for inner-product preservation)
CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server \
    --model-path /path/to/Qwen3-8B/ \
    --kv-cache-dtype turboquant \
    --turboquant-bits 4 \
    --turboquant-mode prod
```

## Accuracy Results

### GSM8K (Qwen3-8B, 4-bit, FlashInfer backend)

| Metric | Value |
|--------|-------|
| Accuracy | 0.890 |
| Invalid | 0.000 |
| Latency | 53.8 s |
| Output throughput | 423 token/s |

### GSM8K (Qwen3-8B, 4-bit, FA3 backend — after fix)

| Metric | Value |
|--------|-------|
| Accuracy | 0.885 |
| Invalid | 0.000 |
| Latency | 84.5 s |
| Output throughput | 1211 token/s |

### NIAH (Qwen3-8B, 4-bit)

| Context Length | Depth 0% | 25% | 50% | 75% | 100% | Average |
|---------------|----------|-----|-----|-----|------|---------|
| 4096 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.000 |
| 8192 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.000 |
| 16384 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.000 |

**Overall recall: 1.000** (paper reference: 0.997 for both TurboQuant 4-bit and full precision)

## Performance Comparison

### TTFT (Time to First Token) — single request, 10K input / 10K output

| Metric | TurboQuant 4-bit | bf16 | Ratio |
|--------|:----------------:|:----:|:-----:|
| Mean TTFT (ms) | 532.12 | 469.84 | 1.13x |
| Median TTFT (ms) | 512.90 | 470.99 | 1.09x |
| P99 TTFT (ms) | 1173.97 | 998.73 | 1.18x |

### Decode Throughput — 100 concurrent requests, ~2.5K input / ~2.4K output

| Metric | TurboQuant 4-bit | bf16 | Ratio |
|--------|:----------------:|:----:|:-----:|
| Output throughput (tok/s) | 1,433 | 1,664 | 0.86x |
| Peak output throughput (tok/s) | 2,500 | 2,980 | 0.84x |
| Total throughput (tok/s) | 2,905 | 3,372 | 0.86x |
| Median TPOT (ms) | 65.93 | 59.20 | 1.11x |
| Median ITL (ms) | 53.17 | 47.13 | 1.13x |

> **Note:** TurboQuant adds ~13% TTFT overhead and ~14% decode throughput reduction for 4-bit mode, in exchange for **3.76x KV cache compression** (73.4% memory savings). The throughput gap is expected to narrow or reverse at longer contexts where KV cache memory becomes the bottleneck.

## KV Precision (4-bit, Qwen3-8B)

| Bits | K cosine | V cosine | K relMSE | V relMSE | K SNR (dB) | V SNR (dB) |
|:----:|:--------:|:--------:|:--------:|:--------:|:----------:|:----------:|
| 2 | 0.9401 | 0.9402 | 0.1163 | 0.1160 | 9.35 | 9.36 |
| 3 | 0.9828 | 0.9829 | 0.0341 | 0.0339 | 14.68 | 14.70 |
| 4 | 0.9953 | 0.9953 | 0.0094 | 0.0093 | 20.29 | 20.32 |

## Compression Efficiency

| Bits | Per-head bytes | bf16 bytes | Compression ratio | Savings |
|:----:|:--------------:|:----------:|:-----------------:|:-------:|
| 2 | 36 | 256 | 7.11x | 85.9% |
| 3 | 52 | 256 | 4.92x | 79.7% |
| 4 | 68 | 256 | 3.76x | 73.4% |

| Context length | bf16 KV | TQ 4-bit total | Saved | Effective ratio |
|:--------------:|:-------:|:--------------:|:-----:|:---------------:|
| 4,096 | 0.56 GB | 0.17 GB | 0.40 GB | 3.41x |
| 32,768 | 4.50 GB | 1.32 GB | 3.18 GB | 3.41x |
| 131,072 | 18.00 GB | 5.28 GB | 12.72 GB | 3.41x |

> Workspace overhead is always ~1 layer of bf16 KV (~2.8% of total for 36-layer models), independent of model depth.

## Benchmark Scripts

Two benchmark scripts are provided under `benchmark/kv_cache/`:

### `bench_turboquant_precision.py` — Precision Testing

Measures KV cache reconstruction accuracy after quantize→dequantize.

```bash
# Direct mode (no model needed, uses random vectors, takes seconds)
CUDA_VISIBLE_DEVICES=6 python benchmark/kv_cache/bench_turboquant_precision.py \
    --bits 4 --bits 3 --bits 2

# E2E mode (loads real model, tests actual KV cache)
CUDA_VISIBLE_DEVICES=6 python benchmark/kv_cache/bench_turboquant_precision.py \
    --model-path /path/to/Qwen3-8B/ --bits 4 --bits 3 --bits 2 --mode e2e

# E2E with minimum prompt token length
CUDA_VISIBLE_DEVICES=6 python benchmark/kv_cache/bench_turboquant_precision.py \
    --model-path /path/to/Qwen3-8B/ --bits 4 --mode e2e --min-tokens 256

# Custom head_dim and token count (direct mode)
CUDA_VISIBLE_DEVICES=6 python benchmark/kv_cache/bench_turboquant_precision.py \
    --bits 4 --head-dim 128 --num-tokens 512
```

**Output (direct mode):**
- Per-bit-width cosine similarity, relMSE, max error, SNR
- Effect of head_dim on precision
- Estimated output distortion after N layers (SNR accumulation)

**Output (e2e mode):**
- Table 1: Overall KV precision per bit-width (real model data)
- Table 2: Per-layer KV precision (which layers are hardest to compress)
- Table 3: KV precision vs token count
- Table 4: Estimated output distortion after N layers

### `bench_turboquant_compression.py` — Compression Efficiency

Calculates memory savings and compression ratios. Pure arithmetic — no GPU needed, only reads model config.

```bash
# Single bit-width
python benchmark/kv_cache/bench_turboquant_compression.py \
    --model-path /path/to/Qwen3-8B/ --bits 4

# Compare multiple bit-widths
python benchmark/kv_cache/bench_turboquant_compression.py \
    --model-path /path/to/Qwen3-8B/ --bits 4 --bits 3 --bits 2

# Prod mode (3-bit MSE + 1-bit QJL)
python benchmark/kv_cache/bench_turboquant_compression.py \
    --model-path /path/to/Qwen3-8B/ --bits 4 --mode prod
```

**Output:**
| Table | Content |
|-------|---------|
| Table 1 | Per-head theoretical compression (independent of token count) |
| Table 2 | Actual GPU memory vs context length (bf16 vs TQ, including workspace) |
| Table 3 | Workspace overhead analysis (fixed ~1 layer of bf16 KV, all context lengths) |
| Table 4 | Net memory savings by context length with workspace percentage |
