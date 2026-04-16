"""
Quantitative precision comparison: TurboQuant vs bf16 KV cache.

Two modes:
  1. --mode direct  (default): Directly apply quantize-dequantize operations to a random vector,
     then calculate the cosine similarity, relative MSE, and maximum error. No model loading is required; the process completes in just a few seconds.

  2. --mode e2e:  Load the full model, perform quantize-dequantize operations on the actual KV cache,
     and calculate the KV reconstruction accuracy (layer-wise and per token length). This requires a single GPU with sufficient memory to accommodate the entire model.

Usage:
  # Quick Test (No Model Required)
  CUDA_VISIBLE_DEVICES=7 python benchmark/kv_cache/bench_turboquant_precision.py --bits 4 --bits 3 --bits 2

  # End-to-End Testing (Requires Model Loading)
  CUDA_VISIBLE_DEVICES=6 python benchmark/kv_cache/bench_turboquant_precision.py \
      --model-path Qwen3-8B/ --bits 4 --bits 3 --bits 2 --mode e2e

  # Customize `head_dim` and number of tokens
  CUDA_VISIBLE_DEVICES=7 python benchmark/kv_cache/bench_turboquant_precision.py \
      --bits 4 --head-dim 128 --num-tokens 512

  # End-to-end testing: Specify the minimum number of prompt tokens.
  CUDA_VISIBLE_DEVICES=6 python benchmark/kv_cache/bench_turboquant_precision.py \
      --model-path Qwen3-8B/ --bits 4 --mode e2e --min-tokens 128
"""

import argparse
import math

import torch
import torch.nn.functional as F


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def relative_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = a.float(), b.float()
    diff = (a_f - b_f).norm().item() ** 2
    ref = a_f.norm().item() ** 2
    return diff / ref if ref > 0 else float("inf")


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def kl_div_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    p = F.log_softmax(logits_a.float(), dim=-1)
    q = F.log_softmax(logits_b.float(), dim=-1)
    return F.kl_div(q, p, log_target=True, reduction="sum").item()


def js_div_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    p = F.softmax(logits_a.float(), dim=-1)
    q = F.softmax(logits_b.float(), dim=-1)
    m = 0.5 * (p + q)
    log_m = torch.log(m + 1e-10)
    kl_pm = F.kl_div(log_m, p, reduction="sum").item()
    kl_qm = F.kl_div(log_m, q, reduction="sum").item()
    return 0.5 * (kl_pm + kl_qm)


def top1_agreement(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    return (logits_a.argmax(dim=-1) == logits_b.argmax(dim=-1)).float().mean().item()


# ── Seeds must match turboquant_memory_pool.py ──
_SEED_K, _SEED_K_LO, _SEED_V, _SEED_V_LO = 42, 43, 137, 138


def make_hadamard_set(head_dim, bits, device):
    from sglang.srt.layers.quantization.turboquant_kernels import (
        HadamardTransform, parse_bits,
    )
    is_mixed, bh, bl = parse_bits(bits)
    if is_mixed:
        split = head_dim // 2
        return {
            "k_hi": HadamardTransform(split, seed=_SEED_K, device=device),
            "k_lo": HadamardTransform(head_dim - split, seed=_SEED_K_LO, device=device),
            "v_hi": HadamardTransform(split, seed=_SEED_V, device=device),
            "v_lo": HadamardTransform(head_dim - split, seed=_SEED_V_LO, device=device),
            "k_split": split, "v_split": split,
        }
    return {
        "k": HadamardTransform(head_dim, seed=_SEED_K, device=device),
        "v": HadamardTransform(head_dim, seed=_SEED_V, device=device),
    }


def quantize_dequantize(flat, hs, bits, mode, which="k"):
    from sglang.srt.layers.quantization.turboquant_kernels import (
        turboquant_quantize, turboquant_dequantize,
        turboquant_quantize_mixed, turboquant_dequantize_mixed,
        parse_bits,
    )
    is_mixed, bh, bl = parse_bits(bits)
    dim = flat.shape[-1]
    if is_mixed:
        h_hi = hs[f"{which}_hi"]
        h_lo = hs[f"{which}_lo"]
        split = hs[f"{which}_split"]
        q = turboquant_quantize_mixed(flat, h_hi, h_lo, bh, bl, split)
        return turboquant_dequantize_mixed(q, h_hi, h_lo, torch.bfloat16)
    else:
        h = hs[which]
        q = turboquant_quantize(flat, h, int(bits), mode)
        return turboquant_dequantize(q, h, int(bits), mode, torch.bfloat16)


def _get_kv_layer(orig_kv, li):
    """Extract K, V from either DynamicCache or legacy tuple cache."""
    if hasattr(orig_kv, 'layers'):
        return orig_kv.layers[li].keys, orig_kv.layers[li].values
    return orig_kv[li]


def _num_cache_layers(orig_kv):
    if hasattr(orig_kv, 'layers'):
        return len(orig_kv.layers)
    return len(orig_kv)


def _quantize_dequantize_kv(k_orig, v_orig, hs, bits, mode):
    """Quantize→dequantize one layer's K/V. Returns (k_recon, v_recon) in original shape."""
    b, h, s, d = k_orig.shape
    _, _, _, dv = v_orig.shape

    k_flat = k_orig.permute(0, 2, 1, 3).reshape(-1, d)
    v_flat = v_orig.permute(0, 2, 1, 3).reshape(-1, dv)

    k_recon = quantize_dequantize(k_flat, hs, bits, mode, "k")[:, :d]
    v_recon = quantize_dequantize(v_flat, hs, bits, mode, "v")[:, :dv]

    k_recon = k_recon.reshape(b, s, h, d).permute(0, 2, 1, 3)
    v_recon = v_recon.reshape(b, s, h, dv).permute(0, 2, 1, 3)
    return k_recon, v_recon


@torch.no_grad()
def run_direct(args):
    """Direct quantize→dequantize precision test. No model needed."""
    from sglang.srt.layers.quantization.turboquant_kernels import (
        initialize_centroids_cache, parse_bits,
    )

    device = torch.device("cuda")
    initialize_centroids_cache(device)

    head_dim = args.head_dim
    num_tokens = args.num_tokens
    num_heads = args.num_kv_heads
    mode = args.tq_mode

    print(f"{'='*95}")
    print(f"  TurboQuant Precision — Direct Quantize/Dequantize")
    print(f"  head_dim={head_dim}  num_tokens={num_tokens}  num_kv_heads={num_heads}  mode={mode}")
    print(f"{'='*95}")

    # Generate realistic KV cache data
    torch.manual_seed(42)
    k_orig = torch.randn(num_tokens, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_orig = torch.randn(num_tokens, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    # ── Per-bit comparison ──
    print(f"\n  {'Bits':>5s}  {'K cos':>8s}  {'V cos':>8s}  "
          f"{'K relMSE':>10s}  {'V relMSE':>10s}  "
          f"{'K maxErr':>10s}  {'V maxErr':>10s}  "
          f"{'K SNR dB':>9s}  {'V SNR dB':>9s}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*9}")

    for bits in sorted(args.bits):
        hs = make_hadamard_set(head_dim, bits, device)
        k_flat = k_orig.reshape(-1, head_dim)
        v_flat = v_orig.reshape(-1, head_dim)
        k_recon = quantize_dequantize(k_flat, hs, bits, mode, "k")[:, :head_dim]
        v_recon = quantize_dequantize(v_flat, hs, bits, mode, "v")[:, :head_dim]
        k_recon = k_recon.reshape(num_tokens, num_heads, head_dim)
        v_recon = v_recon.reshape(num_tokens, num_heads, head_dim)

        k_cos = cosine_sim(k_orig, k_recon)
        v_cos = cosine_sim(v_orig, v_recon)
        k_rmse = relative_mse(k_orig, k_recon)
        v_rmse = relative_mse(v_orig, v_recon)
        k_max = max_abs_error(k_orig, k_recon)
        v_max = max_abs_error(v_orig, v_recon)
        k_snr = 10 * math.log10(1 / k_rmse) if k_rmse > 0 else float("inf")
        v_snr = 10 * math.log10(1 / v_rmse) if v_rmse > 0 else float("inf")

        print(f"  {bits:>5g}  {k_cos:>8.6f}  {v_cos:>8.6f}  "
              f"{k_rmse:>10.6f}  {v_rmse:>10.6f}  "
              f"{k_max:>10.5f}  {v_max:>10.5f}  "
              f"{k_snr:>8.2f}  {v_snr:>8.2f}")

    # ── Per-head-dim comparison ──
    print(f"\n  ── Effect of head_dim on precision (4-bit MSE) ──")
    print(f"  {'head_dim':>9s}  {'K cos':>8s}  {'V cos':>8s}  "
          f"{'K relMSE':>10s}  {'V relMSE':>10s}  {'K SNR dB':>9s}  {'V SNR dB':>9s}")
    print(f"  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*9}")

    bits = 4
    for hd in [64, 96, 128, 256]:
        k_test = torch.randn(256, 8, hd, device=device, dtype=torch.bfloat16)
        v_test = torch.randn(256, 8, hd, device=device, dtype=torch.bfloat16)
        hs = make_hadamard_set(hd, bits, device)
        k_recon = quantize_dequantize(k_test.reshape(-1, hd), hs, bits, mode, "k")[:, :hd].reshape(256, 8, hd)
        v_recon = quantize_dequantize(v_test.reshape(-1, hd), hs, bits, mode, "v")[:, :hd].reshape(256, 8, hd)

        kc = cosine_sim(k_test, k_recon)
        vc = cosine_sim(v_test, v_recon)
        kr = relative_mse(k_test, k_recon)
        vr = relative_mse(v_test, v_recon)
        ksnr = 10 * math.log10(1 / kr) if kr > 0 else float("inf")
        vsnr = 10 * math.log10(1 / vr) if vr > 0 else float("inf")

        print(f"  {hd:>9d}  {kc:>8.6f}  {vc:>8.6f}  {kr:>10.6f}  {vr:>10.6f}  {ksnr:>8.2f}  {vsnr:>8.2f}")

    # ── Simulated logit KL from KV errors ──
    print(f"\n  ── Estimated output distortion after N layers ──")
    print(f"  (assuming per-layer KV errors accumulate independently)")
    print(f"  {'Layers':>7s}", end="")
    for bits in sorted(args.bits):
        print(f"  {'%gb K SNR' % bits:>10s}", end="")
    print()
    print(f"  {'─'*7}", end="")
    for _ in args.bits:
        print(f"  {'─'*10}", end="")
    print()

    num_layers_list = [12, 24, 32, 36, 64, 80]
    for n_layers in num_layers_list:
        print(f"  {n_layers:>7d}", end="")
        for bits in sorted(args.bits):
            hs = make_hadamard_set(head_dim, bits, device)
            k_test = torch.randn(128, 8, head_dim, device=device, dtype=torch.bfloat16)
            k_recon = quantize_dequantize(k_test.reshape(-1, head_dim), hs, bits, mode, "k")[:, :head_dim].reshape(128, 8, head_dim)
            kr = relative_mse(k_test, k_recon)
            est_snr = 10 * math.log10(1 / (kr * math.sqrt(n_layers)))
            print(f"  {est_snr:>9.1f} dB", end="")
        print()

    print(f"\n{'='*95}")


@torch.no_grad()
def run_e2e(args):
    """End-to-end precision test using actual model KV cache data.

    Loads the model, runs forward passes to collect real KV cache values,
    then measures quantize→dequantize precision. Reports:
      1. Overall KV precision per bit-width
      2. Per-layer KV precision (which layers are hardest to compress)
      3. Effect of token count on KV precision
      4. Estimated output distortion after N layers
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sglang.srt.layers.quantization.turboquant_kernels import (
        initialize_centroids_cache, parse_bits,
    )

    device = torch.device("cuda")
    initialize_centroids_cache(device)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)
    num_kv_heads = model.config.num_key_value_heads

    # Natural-language prompts of varying length
    prompts = [
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Italy is Rome. The capital of Spain is Madrid. "
        "The capital of Japan is Tokyo. The capital of China is Beijing. "
        "The capital of Russia is Moscow. The capital of Brazil is Brasilia. "
        "The capital of Australia is Canberra. The capital of Canada is Ottawa. "
        "The capital of India is New Delhi. The capital of Mexico is Mexico City. "
        "The capital of South Korea is Seoul. The capital of Argentina is Buenos Aires. "
        "The capital of Turkey is Ankara. The capital of Egypt is Cairo. "
        "The capital of Thailand is Bangkok. The capital of Indonesia is Jakarta. "
        "The capital of Vietnam is Hanoi. The capital of Poland is Warsaw. "
        "The capital of the Netherlands is Amsterdam. The capital of Sweden is Stockholm. "
        "The capital of Norway is Oslo. The capital of Denmark is Copenhagen. "
        "The capital of Finland is Helsinki. The capital of Switzerland is Bern. "
        "The capital of Austria is Vienna. The capital of Belgium is Brussels. "
        "The capital of Portugal is Lisbon. The capital of Greece is Athens. "
        "The next capital in the list is",

        "Machine learning is a subset of artificial intelligence that focuses on "
        "building systems that learn from data. Deep learning is a further subset "
        "of machine learning that uses neural networks with many layers to model "
        "complex patterns. While traditional machine learning often requires manual "
        "feature engineering, deep learning can automatically learn hierarchical "
        "representations from raw data. Common machine learning algorithms include "
        "decision trees, support vector machines, and random forests. Deep learning "
        "architectures include convolutional neural networks for images, recurrent "
        "neural networks for sequences, and transformer models for language tasks. "
        "Both approaches require training data, but deep learning typically needs "
        "much larger datasets to achieve good performance. The choice between them "
        "depends on the problem complexity, available data, and computational "
        "resources. In practice, simpler machine learning methods often work well "
        "for structured tabular data, while deep learning excels at unstructured "
        "data like images, audio, and natural language. Transfer learning has made "
        "deep learning more accessible by allowing pretrained models to be fine-tuned "
        "on smaller datasets. The key difference is that",

        "The Pythagorean theorem states that in a right triangle, the square of "
        "the length of the hypotenuse equals the sum of the squares of the other "
        "two sides. If the legs have lengths a and b, and the hypotenuse has "
        "length c, then a squared plus b squared equals c squared. This theorem "
        "is fundamental in Euclidean geometry and has numerous practical applications "
        "in construction, navigation, and physics. The theorem can be generalized "
        "to higher dimensions: in a rectangular box with sides a, b, and c, the "
        "space diagonal has length equal to the square root of a squared plus b "
        "squared plus c squared. The converse is also true: if a triangle has "
        "sides satisfying a squared plus b squared equals c squared, then it is "
        "a right triangle. Many proofs exist, including geometric proofs using "
        "area arguments, algebraic proofs, and even proofs attributed to "
        "President Garfield. The theorem extends to the law of cosines for "
        "non-right triangles, where c squared equals a squared plus b squared "
        "minus two times a times b times the cosine of the angle between a and b. "
        "In three-dimensional space, the distance between two points with "
        "coordinates can be found using the three-dimensional version. "
        "The most important application of this theorem is",
    ]

    # Optionally pad prompts to a minimum token length
    if args.min_tokens > 0:
        padded = []
        for p in prompts:
            toks = tokenizer.encode(p)
            if len(toks) < args.min_tokens:
                repeated = p
                while len(tokenizer.encode(repeated)) < args.min_tokens:
                    repeated = repeated + " " + p
                toks = tokenizer.encode(repeated)
                if len(toks) > args.min_tokens:
                    toks = toks[:args.min_tokens]
                repeated = tokenizer.decode(toks)
                padded.append(repeated)
            else:
                padded.append(p)
        prompts = padded

    prompt_lens = [len(tokenizer.encode(p)) for p in prompts]
    print(f"  Prompt token lengths: {prompt_lens} (mean={sum(prompt_lens)/len(prompt_lens):.0f})")

    print(f"\n{'='*95}")
    print(f"  TurboQuant Precision — End-to-End (Real Model KV Cache)")
    print(f"  {num_layers} layers  |  {num_kv_heads} KV heads  |  head_dim={head_dim}")
    print(f"{'='*95}")

    # Collect KV cache from the first prompt (used for Tables 1 & 2)
    prompt = prompts[0]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    orig_kv = out.past_key_values
    num_tokens = inputs["input_ids"].shape[-1]
    del out

    # ── Table 1: Overall KV precision per bit-width ──
    print(f"\n  ── Table 1: KV Reconstruction Precision (real model, {num_tokens} tokens) ──")
    print(f"  {'Bits':>5s}  {'K cos':>8s}  {'V cos':>8s}  "
          f"{'K relMSE':>10s}  {'V relMSE':>10s}  "
          f"{'K SNR dB':>9s}  {'V SNR dB':>9s}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*9}")

    for bits in sorted(args.bits):
        hs = make_hadamard_set(head_dim, bits, device)
        all_k_cos, all_v_cos, all_k_rmse, all_v_rmse = [], [], [], []

        for li in range(num_layers):
            k_orig, v_orig = _get_kv_layer(orig_kv, li)
            k_recon, v_recon = _quantize_dequantize_kv(k_orig, v_orig, hs, bits, args.tq_mode)
            all_k_cos.append(cosine_sim(k_orig, k_recon))
            all_v_cos.append(cosine_sim(v_orig, v_recon))
            all_k_rmse.append(relative_mse(k_orig, k_recon))
            all_v_rmse.append(relative_mse(v_orig, v_recon))

        k_cos = sum(all_k_cos) / len(all_k_cos)
        v_cos = sum(all_v_cos) / len(all_v_cos)
        k_rmse = sum(all_k_rmse) / len(all_k_rmse)
        v_rmse = sum(all_v_rmse) / len(all_v_rmse)
        k_snr = 10 * math.log10(1 / k_rmse) if k_rmse > 0 else float("inf")
        v_snr = 10 * math.log10(1 / v_rmse) if v_rmse > 0 else float("inf")

        print(f"  {bits:>5g}  {k_cos:>8.6f}  {v_cos:>8.6f}  "
              f"{k_rmse:>10.6f}  {v_rmse:>10.6f}  {k_snr:>8.2f}  {v_snr:>8.2f}")

    # ── Table 2: Per-layer KV precision (4-bit) ──
    ref_bits = 4
    hs = make_hadamard_set(head_dim, ref_bits, device)
    print(f"\n  ── Table 2: Per-Layer Precision ({ref_bits}-bit, {num_tokens} tokens) ──")
    print(f"  {'Layer':>5s}  {'K cos':>8s}  {'V cos':>8s}  "
          f"{'K relMSE':>10s}  {'V relMSE':>10s}  {'K SNR dB':>9s}  {'V SNR dB':>9s}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*9}")

    for li in range(num_layers):
        k_orig, v_orig = _get_kv_layer(orig_kv, li)
        k_recon, v_recon = _quantize_dequantize_kv(k_orig, v_orig, hs, ref_bits, args.tq_mode)
        kc = cosine_sim(k_orig, k_recon)
        vc = cosine_sim(v_orig, v_recon)
        kr = relative_mse(k_orig, k_recon)
        vr = relative_mse(v_orig, v_recon)
        ksnr = 10 * math.log10(1 / kr) if kr > 0 else float("inf")
        vsnr = 10 * math.log10(1 / vr) if vr > 0 else float("inf")

        if li < 5 or li >= num_layers - 2 or li % 9 == 0:
            print(f"  {li:>5d}  {kc:>8.6f}  {vc:>8.6f}  "
                  f"{kr:>10.6f}  {vr:>10.6f}  {ksnr:>8.2f}  {vsnr:>8.2f}")
        elif li == 5:
            print(f"  {'...':>5s}  {'...':>8s}  {'...':>8s}  "
                  f"{'...':>10s}  {'...':>10s}  {'...':>9s}  {'...':>9s}")

    del orig_kv

    # ── Table 3: Effect of token count on KV precision ──
    print(f"\n  ── Table 3: KV Precision vs Token Count ({ref_bits}-bit) ──")
    print(f"  {'Tokens':>7s}  {'K cos':>8s}  {'V cos':>8s}  "
          f"{'K relMSE':>10s}  {'V relMSE':>10s}  {'K SNR dB':>9s}  {'V SNR dB':>9s}")
    print(f"  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*9}")

    for prompt_text in prompts:
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        kv = out.past_key_values
        n_toks = inputs["input_ids"].shape[-1]
        del out

        all_k_cos, all_v_cos, all_k_rmse, all_v_rmse = [], [], [], []
        for li in range(num_layers):
            k_orig, v_orig = _get_kv_layer(kv, li)
            k_recon, v_recon = _quantize_dequantize_kv(k_orig, v_orig, hs, ref_bits, args.tq_mode)
            all_k_cos.append(cosine_sim(k_orig, k_recon))
            all_v_cos.append(cosine_sim(v_orig, v_recon))
            all_k_rmse.append(relative_mse(k_orig, k_recon))
            all_v_rmse.append(relative_mse(v_orig, v_recon))

        kc = sum(all_k_cos) / len(all_k_cos)
        vc = sum(all_v_cos) / len(all_v_cos)
        kr = sum(all_k_rmse) / len(all_k_rmse)
        vr = sum(all_v_rmse) / len(all_v_rmse)
        ksnr = 10 * math.log10(1 / kr) if kr > 0 else float("inf")
        vsnr = 10 * math.log10(1 / vr) if vr > 0 else float("inf")

        print(f"  {n_toks:>7d}  {kc:>8.6f}  {vc:>8.6f}  "
              f"{kr:>10.6f}  {vr:>10.6f}  {ksnr:>8.2f}  {vsnr:>8.2f}")

        del kv
        torch.cuda.empty_cache()

    # ── Table 4: Estimated output distortion ──
    print(f"\n  ── Table 4: Estimated Output Distortion After {num_layers} Layers ──")
    print(f"  (KV errors accumulate as sqrt(N) × per-layer relMSE)")
    print(f"  {'Bits':>5s}  {'Per-layer':>10s}  {'Est. output':>12s}  {'Est. logit':>12s}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*12}")

    for bits in sorted(args.bits):
        hs = make_hadamard_set(head_dim, bits, device)
        # Use a fresh forward pass for the estimate
        inputs = tokenizer(prompts[0], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        kv = out.past_key_values
        del out

        all_k_rmse = []
        for li in range(num_layers):
            k_orig, _ = _get_kv_layer(kv, li)
            k_recon, _ = _quantize_dequantize_kv(k_orig, _get_kv_layer(kv, li)[1], hs, bits, args.tq_mode)
            all_k_rmse.append(relative_mse(k_orig, k_recon))
        del kv

        avg_rmse = sum(all_k_rmse) / len(all_k_rmse)
        est_output_rmse = avg_rmse * math.sqrt(num_layers)
        est_output_snr = 10 * math.log10(1 / est_output_rmse) if est_output_rmse > 0 else float("inf")
        est_logit_cos = 1.0 / (1.0 + 8.0 * math.sqrt(num_layers) * avg_rmse)

        print(f"  {bits:>5g}  {avg_rmse:>10.6f}  {est_output_snr:>10.2f} dB  {est_logit_cos:>11.4f}")

    del model
    torch.cuda.empty_cache()
    print(f"\n{'='*95}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None,
                        help="Model path (required for --mode e2e)")
    parser.add_argument("--bits", type=float, action="append", default=[],
                        help="Bit-width(s) to evaluate")
    parser.add_argument("--tq-mode", default="mse", choices=["mse", "prod"],
                        dest="tq_mode", help="TurboQuant mode")
    parser.add_argument("--mode", default="direct", choices=["direct", "e2e"],
                        help="direct=no model, e2e=real model KV comparison")

    # Direct mode params
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--num-kv-heads", type=int, default=8)

    # E2E mode params
    parser.add_argument("--min-tokens", type=int, default=0,
                        help="Pad each prompt to at least this many tokens (e2e mode)")
    args = parser.parse_args()

    if not args.bits:
        args.bits = [4]

    if args.mode == "e2e":
        if args.model_path is None:
            parser.error("--model-path is required for --mode e2e")
        run_e2e(args)
    else:
        run_direct(args)


if __name__ == "__main__":
    main()
