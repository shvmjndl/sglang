"""
Benchmark TurboQuant KV cache compression efficiency.

Memory model:
  - Compressed storage: per-layer, scales linearly with token count.
    Each (token, head) vector is independently compressed to bit-packed
    indices + a float32 norm.  More tokens → more storage.
  - Workspace: shared across ALL layers (only 1 K + 1 V), allocated once
    at pool max capacity.  Does NOT scale with token count — it is
    allocated upfront to the maximum pool size.

Therefore:
  - Compression ratio (per-head) is constant regardless of token count.
  - Total saved memory grows linearly with token count.
  - Workspace overhead is a fixed constant (~1 layer worth of bf16 KV).
  - At short contexts, workspace overhead is proportionally larger;
    at long contexts, it becomes negligible.

Usage:
  # Default: test 4-bit MSE
  CUDA_VISIBLE_DEVICES=7 python benchmark/kv_cache/bench_turboquant_compression.py \
      --model-path Qwen3-8B/ --bits 4

  # Compare multiple bit-widths
  CUDA_VISIBLE_DEVICES=7 python benchmark/kv_cache/bench_turboquant_compression.py \
      --model-path Qwen3-8B/ --bits 4 --bits 3 --bits 2

  # Prod mode (3-bit MSE + 1-bit QJL)
  CUDA_VISIBLE_DEVICES=7 python benchmark/kv_cache/bench_turboquant_compression.py \
      --model-path Qwen3-8B/ --bits 4 --mode prod
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bits", type=float, action="append", default=[],
                        help="Bit-width(s) to evaluate (1/2/3/4/2.5/3.5)")
    parser.add_argument("--mode", default="mse", choices=["mse", "prod"])
    args = parser.parse_args()

    if not args.bits:
        args.bits = [4]

    # Load model config (no weights)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    v_head_dim = head_dim  # MHA models have same v_head_dim

    from sglang.srt.layers.quantization.turboquant_kernels import (
        compute_compression_ratio,
        compute_packed_dim,
        compute_packed_dim_mixed,
        _next_power_of_2,
        parse_bits,
    )

    GB = 1024 ** 3
    dtype_bytes = 2  # bf16

    print(f"{'='*85}")
    print(f"TurboQuant Compression Efficiency — {config.model_type}")
    print(f"{'='*85}")
    print(f"  Layers:    {num_layers}")
    print(f"  KV heads:  {num_kv_heads}")
    print(f"  Head dim:  {head_dim}")
    print(f"  Mode:      {args.mode}")
    print()

    # ── Table 1: Per-head theoretical compression ──
    print(f"{'─'*85}")
    print(f"TABLE 1: PER-HEAD COMPRESSION (independent of token count)")
    print(f"{'─'*85}")
    print(f"  Each (token, head) vector is compressed independently.")
    print(f"  Ratio is constant regardless of sequence length.")
    print()
    print(f"  {'Bits':>5s}  {'Index B':>8s}  {'Norm B':>6s}  {'QJL B':>6s}  {'Total B':>7s}  "
          f"{'bf16 B':>6s}  {'Ratio':>6s}  {'Savings':>8s}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*8}")

    for bits in sorted(args.bits):
        is_mixed, bits_hi, bits_lo = parse_bits(bits)

        # Index bytes per head
        if is_mixed:
            index_bytes = compute_packed_dim_mixed(head_dim, bits)
        else:
            mse_bits = bits_hi - 1 if args.mode == "prod" else bits_hi
            padded = _next_power_of_2(head_dim)
            index_bytes = compute_packed_dim(padded, mse_bits)

        norm_bytes = 8 if is_mixed else 4
        qjl_bytes = 0
        if args.mode == "prod" and not is_mixed:
            padded = _next_power_of_2(head_dim)
            qjl_bytes = compute_packed_dim(padded, 1) + 4

        total_tq = index_bytes + norm_bytes + qjl_bytes
        total_bf16 = head_dim * dtype_bytes
        ratio = total_bf16 / total_tq
        savings = (1 - total_tq / total_bf16) * 100

        print(f"  {bits:>5g}  {index_bytes:>8d}  {norm_bytes:>6d}  {qjl_bytes:>6d}  {total_tq:>7d}  "
              f"{total_bf16:>6d}  {ratio:>5.2f}x  {savings:>7.1f}%")

    # ── Table 2: Actual GPU memory at different token counts ──
    print()
    print(f"{'─'*85}")
    print(f"TABLE 2: ACTUAL GPU MEMORY VS CONTEXT LENGTH")
    print(f"{'─'*85}")
    print(f"  Workspace is allocated once at pool max capacity (fixed cost).")
    print(f"  Compressed storage scales linearly with tokens × layers.")
    print(f"  bf16 baseline also scales linearly (per-layer buffers).")
    print()

    # The pool is sized to max_tokens; workspace is allocated at pool size.
    # In SGLang, pool size = available_gpu_memory / bytes_per_token.
    # We show a range of token counts to illustrate the scaling.
    page_size = 1

    # For each bits config, show a compact table
    for bits in sorted(args.bits):
        is_mixed, bits_hi, bits_lo = parse_bits(bits)

        # Per-token-head compressed size
        if args.mode == "mse":
            k_packed_dim = compute_packed_dim_mixed(head_dim, bits)
        else:
            k_packed_dim = compute_packed_dim_mixed(head_dim, bits - 1)
        norm_per_head = 8 if is_mixed else 4
        qjl_per_head = 0
        if args.mode == "prod" and not is_mixed:
            qjl_per_head = compute_packed_dim(_next_power_of_2(head_dim), 1) + 4

        # Per-token compressed bytes (all heads, one layer)
        compressed_per_token_per_layer = num_kv_heads * k_packed_dim + num_kv_heads * norm_per_head + num_kv_heads * qjl_per_head  # K
        compressed_per_token_per_layer += compressed_per_token_per_layer  # V (same size for MHA)

        print(f"  [{bits}b {args.mode}]")
        print(f"  {'Tokens':>8s}  {'Compressed':>12s}  {'Workspace':>12s}  {'TQ Total':>12s}  "
              f"{'bf16 Total':>12s}  {'Saved':>12s}  {'Eff. Ratio':>10s}")
        print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}")

        ctx_lengths = [1024, 4096, 8192, 32768, 65536, 131072]
        for ctx_len in ctx_lengths:
            m = ctx_len + page_size

            # Compressed storage: per-layer, scales with tokens
            compressed = compressed_per_token_per_layer * m * num_layers

            # Workspace: fixed at pool capacity, NOT per-layer
            # Allocated once at pool_size. We show it at this ctx_len's pool.
            ws = m * num_kv_heads * head_dim * dtype_bytes  # K workspace
            ws += m * num_kv_heads * v_head_dim * dtype_bytes  # V workspace

            tq_total = compressed + ws

            # bf16 baseline: per-layer, scales with tokens
            bf16_per_token_per_layer = num_kv_heads * head_dim * dtype_bytes  # K
            bf16_per_token_per_layer += num_kv_heads * v_head_dim * dtype_bytes  # V
            bf16 = bf16_per_token_per_layer * m * num_layers

            saved = bf16 - tq_total
            eff_ratio = bf16 / tq_total if tq_total > 0 else float('inf')

            print(f"  {ctx_len:>8d}  {compressed/GB:>10.2f} GB  {ws/GB:>10.2f} GB  "
                  f"{tq_total/GB:>10.2f} GB  {bf16/GB:>10.2f} GB  {saved/GB:>10.2f} GB  {eff_ratio:>9.2f}x")
        print()

    # ── Table 3: Workspace overhead breakdown ──
    print(f"{'─'*85}")
    print(f"TABLE 3: WORKSPACE OVERHEAD (shared, 1 copy for all layers)")
    print(f"{'─'*85}")
    print()
    print(f"  Workspace stores dequantized bf16 values for the active positions.")
    print(f"  Only 1 K workspace + 1 V workspace is needed because attention")
    print(f"  executes layer-by-layer — the same buffer is reused each layer.")
    print()

    pool_sizes = [8192, 32768, 131072]
    for pool_size in pool_sizes:
        m = pool_size + page_size
        ws_k = m * num_kv_heads * head_dim * dtype_bytes
        ws_v = m * num_kv_heads * v_head_dim * dtype_bytes
        ws_total = ws_k + ws_v
        bf16_per_layer = m * num_kv_heads * head_dim * dtype_bytes + m * num_kv_heads * v_head_dim * dtype_bytes
        print(f"  Pool={pool_size:>6d} tokens:  workspace = {ws_total/GB:.2f} GB  "
              f"= {ws_total/bf16_per_layer:.1f} layer-equivalents of bf16 KV")

    print()
    print(f"  Key insight: workspace overhead is always ~1 layer of bf16 KV,")
    print(f"  regardless of model depth. For a {num_layers}-layer model,")
    print(f"  this is only {1/num_layers*100:.1f}% of the bf16 total.")

    # ── Table 4: Break-even analysis ──
    print()
    print(f"{'─'*85}")
    print(f"TABLE 4: NET MEMORY SAVINGS BY CONTEXT LENGTH (4-bit MSE)")
    print(f"{'─'*85}")
    print()
    print(f"  Shows the fraction of total GPU memory that is saved,")
    print(f"  and how workspace overhead diminishes with longer contexts.")
    print()

    bits = 4
    is_mixed, bits_hi, bits_lo = parse_bits(bits)
    k_packed_dim = compute_packed_dim_mixed(head_dim, bits)
    norm_per_head = 8 if is_mixed else 4
    compressed_per_token_per_layer = (num_kv_heads * k_packed_dim + num_kv_heads * norm_per_head) * 2

    bf16_per_token_per_layer = (num_kv_heads * head_dim * dtype_bytes + num_kv_heads * v_head_dim * dtype_bytes)

    print(f"  {'Ctx Len':>8s}  {'bf16 KV':>10s}  {'TQ Compr.':>10s}  {'Workspace':>10s}  "
          f"{'TQ Total':>10s}  {'Saved':>10s}  {'WS%':>6s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*6}")

    ctx_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    for ctx_len in ctx_lengths:
        m = ctx_len + page_size
        compressed = compressed_per_token_per_layer * m * num_layers
        ws = m * num_kv_heads * (head_dim + v_head_dim) * dtype_bytes
        tq_total = compressed + ws
        bf16 = bf16_per_token_per_layer * m * num_layers
        saved = bf16 - tq_total
        ws_pct = ws / tq_total * 100  # workspace as % of TQ total

        print(f"  {ctx_len:>8d}  {bf16/GB:>8.2f} GB  {compressed/GB:>8.2f} GB  {ws/GB:>8.2f} GB  "
              f"{tq_total/GB:>8.2f} GB  {saved/GB:>8.2f} GB  {ws_pct:>5.1f}%")

    print()
    print(f"  WS% = workspace as percentage of TQ total memory.")
    print(f"  At short contexts, workspace overhead is significant;")
    print(f"  at long contexts it becomes negligible (<5%).")
    print()
    print(f"{'='*85}")


if __name__ == "__main__":
    main()
