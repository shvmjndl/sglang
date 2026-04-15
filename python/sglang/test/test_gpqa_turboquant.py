"""
TurboQuant benchmark: Qwen3-8B on GPQA (Diamond) dataset.

Two modes:
  - baseline (bf16): standard greedy generation
  - turboquant: KV cache compressed at specified bit-width

Usage:
  python test_gpqa_qwen3_8b.py                    # run both modes, default 4-bit TQ
  python test_gpqa_qwen3_8b.py --bits 2.5         # specify TQ bit-width
  python test_gpqa_qwen3_8b.py --mode baseline    # baseline only
  python test_gpqa_qwen3_8b.py --mode turboquant  # TQ only
  python test_gpqa_qwen3_8b.py --max-samples 50   # limit sample count
"""

import argparse
import gc
import importlib.util
import json
import os
import re
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Import TurboQuant kernels (same approach as the existing test file)
# ---------------------------------------------------------------------------
_kernels_path = os.path.join(
    os.path.dirname(__file__),
    "..", "srt", "layers", "quantization", "turboquant_kernels.py",
)
_spec = importlib.util.spec_from_file_location(
    "turboquant_kernels", os.path.abspath(_kernels_path)
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

HadamardTransform = _mod.HadamardTransform
parse_bits = _mod.parse_bits
compute_compression_ratio = _mod.compute_compression_ratio
turboquant_quantize = _mod.turboquant_quantize
turboquant_dequantize = _mod.turboquant_dequantize
turboquant_quantize_mixed = _mod.turboquant_quantize_mixed
turboquant_dequantize_mixed = _mod.turboquant_dequantize_mixed

DEVICE = torch.device("cuda")

# Hadamard seeds (must match turboquant_memory_pool.py)
_SEED_K, _SEED_K_LO, _SEED_V, _SEED_V_LO = 42, 43, 137, 138

rotate_matrix_S=torch.randn(128,128)
# ---------------------------------------------------------------------------
# TurboQuant helpers (reused from existing test)
# ---------------------------------------------------------------------------

def _make_hadamard_set(hd, bits):
    is_mixed, bh, bl = parse_bits(bits)
    if is_mixed:
        split = hd // 2
        return {
            "k_h": None, "v_h": None,
            "k_hi": HadamardTransform(split, seed=_SEED_K, device=DEVICE),
            "k_lo": HadamardTransform(hd - split, seed=_SEED_K_LO, device=DEVICE),
            "v_hi": HadamardTransform(split, seed=_SEED_V, device=DEVICE),
            "v_lo": HadamardTransform(hd - split, seed=_SEED_V_LO, device=DEVICE),
            "k_split": split, "v_split": split,
        }
    return {
        "k_h": HadamardTransform(hd, seed=_SEED_K, device=DEVICE),
        "v_h": HadamardTransform(hd, seed=_SEED_V, device=DEVICE),
    }


def _quantize_roundtrip(flat, bits, hs, is_key=True,quant_type=None):
    is_mixed, bh, bl = parse_bits(bits)
    if is_mixed:
        hi = hs["k_hi"] if is_key else hs["v_hi"]
        lo = hs["k_lo"] if is_key else hs["v_lo"]
        sp = hs["k_split"] if is_key else hs["v_split"]
        q = turboquant_quantize_mixed(flat, hi, lo, bh, bl, sp)
        return turboquant_dequantize_mixed(q, hi, lo, torch.bfloat16)
    h = hs["k_h"] if is_key else hs["v_h"]
    q = turboquant_quantize(flat, h, int(bits), quant_type)
    return turboquant_dequantize(q, h, int(bits), quant_type, torch.bfloat16)

def _tq_generate_batch(model, tokenizer, inputs, bits, hd, max_new=64,quant_type=None):
    """
    Batched autoregressive generation with TQ-compressed KV cache.

    Args:
        inputs: tokenizer output dict with input_ids, attention_mask (batch, seq_len)
        bits: TQ bit-width
        hd: head dimension
        max_new: max new tokens to generate

    Returns:
        list of lists: generated token ids per sample (excluding input tokens)
    """
    from transformers import DynamicCache

    hs = _make_hadamard_set(hd, bits)
    input_ids = inputs["input_ids"]                # (B, S)
    attention_mask = inputs["attention_mask"]       # (B, S)
    bs = input_ids.shape[0]

    # Track which sequences have finished (hit EOS)
    finished = torch.zeros(bs, dtype=torch.bool, device=DEVICE)
    # Store generated token ids per sample
    gen_ids = [[] for _ in range(bs)]

    with torch.no_grad():
        # --- Prefill ---
        out = model(**inputs, use_cache=True)

        # Quantize the prefill KV cache
        tql = []
        for lkv in out.past_key_values:
            k, v = lkv[0], lkv[1]  # (B, H, S, D)
            b, h, s, dk = k.shape
            dv = v.shape[-1]
            kr = _quantize_roundtrip(
                k.permute(0, 2, 1, 3).reshape(-1, dk), bits, hs, True,quant_type
            )[:, :dk].reshape(b, s, h, dk).permute(0, 2, 1, 3)
            vr = _quantize_roundtrip(
                v.permute(0, 2, 1, 3).reshape(-1, dv), bits, hs, False,quant_type
            )[:, :dv].reshape(b, s, h, dv).permute(0, 2, 1, 3)
            tql.append((kr, vr))

        tc = DynamicCache()
        for li, (kt, vt) in enumerate(tql):
            tc.update(kt.contiguous(), vt.contiguous(), li)

        # First generated token: pick from last non-pad position per sample
        # For left-padded inputs, the last position is always valid
        nt = out.logits[:, -1:, :].argmax(dim=-1)  # (B, 1)

        # Update attention mask for the new token
        cur_mask = torch.cat(
            [attention_mask, torch.ones(bs, 1, dtype=attention_mask.dtype, device=DEVICE)],
            dim=1,
        )

        for i in range(bs):
            tok = nt[i, 0].item()
            if not finished[i]:
                gen_ids[i].append(tok)
                if tok == tokenizer.eos_token_id:
                    finished[i] = True

        # --- Decode loop ---
        for step in range(max_new - 1):
            if finished.all():
                break

            out = model(
                input_ids=nt,
                attention_mask=cur_mask,
                past_key_values=tc,
                use_cache=True,
            )
            tc = out.past_key_values

            # Quantize only the newly appended KV token
            nl = []
            for lkv in tc:
                kf, vf = lkv[0], lkv[1]            # (B, H, S_total, D)
                kn = kf[:, :, -1:, :]               # (B, H, 1, D)
                vn = vf[:, :, -1:, :]
                b2, h2, _, dk2 = kn.shape
                dv2 = vn.shape[-1]
                kr2 = _quantize_roundtrip(
                    kn.permute(0, 2, 1, 3).reshape(-1, dk2), bits, hs, True,quant_type
                )[:, :dk2].reshape(b2, 1, h2, dk2).permute(0, 2, 1, 3)
                vr2 = _quantize_roundtrip(
                    vn.permute(0, 2, 1, 3).reshape(-1, dv2), bits, hs, False,quant_type
                )[:, :dv2].reshape(b2, 1, h2, dv2).permute(0, 2, 1, 3)
                nl.append((
                    torch.cat([kf[:, :, :-1, :], kr2], dim=2),
                    torch.cat([vf[:, :, :-1, :], vr2], dim=2),
                ))

            tc = DynamicCache()
            for li, (kt, vt) in enumerate(nl):
                tc.update(kt.contiguous(), vt.contiguous(), li)

            nt = out.logits[:, -1:, :].argmax(dim=-1)  # (B, 1)

            # Extend attention mask
            cur_mask = torch.cat(
                [cur_mask, torch.ones(bs, 1, dtype=cur_mask.dtype, device=DEVICE)],
                dim=1,
            )

            # Collect tokens, check EOS
            for i in range(bs):
                tok = nt[i, 0].item()
                if not finished[i]:
                    gen_ids[i].append(tok)
                    if tok == tokenizer.eos_token_id:
                        finished[i] = True

    return gen_ids

def _tq_generate(model, tokenizer, inputs, bits, hd, max_new=64):
    """Autoregressive generation with TQ-compressed KV cache."""
    from transformers import DynamicCache
    hs = _make_hadamard_set(hd, bits)

    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        tql = []
        for lkv in out.past_key_values:
            k, v = lkv[0], lkv[1]
            b, h, s, dk = k.shape
            dv = v.shape[-1]
            kr = _quantize_roundtrip(
                k.permute(0, 2, 1, 3).reshape(-1, dk), bits, hs, True
            )[:, :dk].reshape(b, s, h, dk).permute(0, 2, 1, 3)
            vr = _quantize_roundtrip(
                v.permute(0, 2, 1, 3).reshape(-1, dv), bits, hs, False
            )[:, :dv].reshape(b, s, h, dv).permute(0, 2, 1, 3)
            tql.append((kr, vr))

        tc = DynamicCache()
        for li, (kt, vt) in enumerate(tql):
            tc.update(kt.contiguous(), vt.contiguous(), li)

        nt = out.logits[:, -1:].argmax(dim=-1)
        gen = [nt.item()]

        for _ in range(max_new - 1):
            out = model(nt, past_key_values=tc, use_cache=True)
            tc = out.past_key_values
            nl = []
            for lkv in tc:
                kf, vf = lkv[0], lkv[1]
                kn, vn = kf[:, :, -1:, :], vf[:, :, -1:, :]
                b2, h2, _, dk2 = kn.shape
                dv2 = vn.shape[-1]
                kr2 = _quantize_roundtrip(
                    kn.permute(0, 2, 1, 3).reshape(-1, dk2), bits, hs, True
                )[:, :dk2].reshape(b2, 1, h2, dk2).permute(0, 2, 1, 3)
                vr2 = _quantize_roundtrip(
                    vn.permute(0, 2, 1, 3).reshape(-1, dv2), bits, hs, False
                )[:, :dv2].reshape(b2, 1, h2, dv2).permute(0, 2, 1, 3)
                nl.append((
                    torch.cat([kf[:, :, :-1, :], kr2], dim=2),
                    torch.cat([vf[:, :, :-1, :], vr2], dim=2),
                ))
            tc = DynamicCache()
            for li, (kt, vt) in enumerate(nl):
                tc.update(kt.contiguous(), vt.contiguous(), li)
            nt = out.logits[:, -1:].argmax(dim=-1)
            gen.append(nt.item())
            if nt.item() == tokenizer.eos_token_id:
                break
    return gen


# ---------------------------------------------------------------------------
# GPQA dataset loading
# ---------------------------------------------------------------------------

def load_gpqa_dataset(split="train", max_samples=None, shard=None):
    """
    Load GPQA Diamond from HuggingFace datasets.
 
    Args:
        split: dataset split
        max_samples: cap total samples before sharding
        shard: if set (1-4), return only that quarter of the dataset
 
    Each sample has:
      - question (str)
      - choices: list of 4 answer strings
      - answer_index: int (0-3), the correct answer
    """
    from datasets import load_dataset
    import hashlib
    import random as _random
 
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=split)
 
    samples = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
 
        # GPQA Diamond fields:
        #   "Question", "Correct Answer", "Incorrect Answer 1/2/3"
        question = row["Question"]
        correct = row["Correct Answer"]
        incorrects = [
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
 
        # Deterministic shuffle: place correct answer at a fixed position
        # based on hash to avoid position bias but keep reproducibility
        choices = incorrects + [correct]
        seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
        rng = _random.Random(seed)
        rng.shuffle(choices)
        answer_index = choices.index(correct)
 
        samples.append({
            "question": question,
            "choices": choices,
            "answer_index": answer_index,
        })
 
    # Apply sharding: split into 4 equal parts, return the requested quarter
    if shard is not None:
        assert 1 <= shard <= 4, f"shard must be 1-4, got {shard}"
        total = len(samples)
        shard_size = (total + 3) // 4  # ceil division
        start = (shard - 1) * shard_size
        end = min(shard * shard_size, total)
        samples = samples[start:end]
 
    return samples


# ---------------------------------------------------------------------------
# Prompt formatting for GPQA multiple-choice
# ---------------------------------------------------------------------------
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

CHOICE_LABELS = ["A", "B", "C", "D"]



def format_question_block(sample):
    """Format the question + choices block."""
    prompt = QUERY_TEMPLATE_MULTICHOICE.format(
        Question=sample['question'],
        A=sample["choices"][0],
        B=sample["choices"][1],
        C=sample["choices"][2],
        D=sample["choices"][3]
    )

    return prompt


def extract_answer(generated_text):
    """
    Extract the predicted answer letter from generated text.
    Returns one of 'A', 'B', 'C', 'D' or None.
    """
    text = generated_text.strip()


    # Try to find any letter in the text
    answer_pattern=r"(?i)Answer\s*:\s*([A-D])"

    matches = re.findall(answer_pattern, text)
    extracted_answer = matches[-1].upper() if matches else None

    return extracted_answer


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------
def evaluate_gpqa(
    model,
    tokenizer,
    samples,
    mode="baseline",
    bits=4,
    hd=128,
    max_new_tokens=32768,
    batch_size=8,
    verbose=True,
    quant_type=None
):
    """
    Evaluate model on GPQA samples with batch processing.

    Args:
        mode: "baseline" (bf16) or "turboquant"
        bits: TQ bit-width (used only when mode="turboquant")
        hd: head dimension
        max_new_tokens: max tokens to generate per sample
        batch_size: number of samples per batch
    Returns:
        dict with accuracy, per-sample results, timing
    """
    correct = 0
    total = 0
    results = []
    t0 = time.time()

    # Left-padding is required for batched generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_batches = (len(samples) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(samples))
        batch_samples = samples[start:end]
        cur_bs = len(batch_samples)

        # Format prompts for the batch
        # prompts = [format_gpqa_prompt(s) for s in batch_samples]
        # inputs = tokenizer(
        #     prompts, return_tensors="pt", padding=True, truncation=True
        # ).to(DEVICE)
        texts = []
        for s in batch_samples:
            messages = [
                {
                    "role": "user",
                    "content": format_question_block(s)
                    # + "\nAnswer with the letter A, B, C, or D."
                },
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            # print(f"zsp get prompt{text}")
            texts.append(text)

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        if mode == "baseline":
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            # Extract generated tokens per sample (skip input portion)
            input_len = inputs["input_ids"].shape[1]
            for i in range(cur_bs):
                gen_ids = output_ids[i, input_len:].tolist()
                # Strip pad tokens
                gen_ids = [t for t in gen_ids if t != tokenizer.pad_token_id]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                predicted = extract_answer(gen_text)
                expected = CHOICE_LABELS[batch_samples[i]["answer_index"]]
                is_correct = predicted == expected
                if is_correct:
                    correct += 1
                total += 1
                results.append({
                    "index": start + i,
                    "predicted": predicted,
                    "expected": expected,
                    "correct": is_correct,
                    "generated_text": gen_text.strip()[:100],
                })

        elif mode == "turboquant":
            batch_gen_ids = _tq_generate_batch(
                model, tokenizer, inputs, bits, hd, max_new=max_new_tokens,quant_type=quant_type
            )
            for i in range(cur_bs):
                gen_ids = batch_gen_ids[i]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                # print(f"zsp get gen text {gen_text}")
                predicted = extract_answer(gen_text)
                expected = CHOICE_LABELS[batch_samples[i]["answer_index"]]
                print(f"zsp get pred {predicted} expected {expected}")
                is_correct = predicted == expected
                if is_correct:
                    correct += 1
                total += 1
                results.append({
                    "index": start + i,
                    "predicted": predicted,
                    "expected": expected,
                    "correct": is_correct,
                    "generated_text": gen_text.strip()[:100],
                })
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if verbose:
            processed = end
            acc_so_far = correct / total if total > 0 else 0
            print(
                f"    [{mode}] batch {batch_idx+1}/{num_batches} "
                f"({processed}/{len(samples)}): "
                f"acc={acc_so_far:.1%} ({correct}/{total})"
            )

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    elapsed = time.time() - t0
    accuracy = correct / total if total > 0 else 0.0

    return {
        "mode": mode,
        "bits": bits if mode == "turboquant" else "bf16",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "elapsed_s": elapsed,
        "results": results,
    }



# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------

def test_gpqa_qwen3_8b(
    bits_list=None,
    max_samples=None,
    modes=None,
    shard=None,
    quant_type=None
):
    """
    Run GPQA Diamond benchmark on Qwen3-8B with baseline and TurboQuant.

    Args:
        bits_list: list of TQ bit-widths to test (default: [4])
        max_samples: limit number of GPQA samples (None = all)
        modes: list of modes to run, subset of ["baseline", "turboquant"]
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if bits_list is None:
        bits_list = [4]
    if modes is None:
        modes = ["baseline", "turboquant"]

    model_id = "Qwen/Qwen3-8B"

    print(f"{'='*70}")
    print(f"GPQA Diamond Benchmark: {model_id}")
    print(f"  Modes: {modes}")
    print(f"  TQ bit-widths: {bits_list}")
    print(f"  Max samples: {max_samples or 'all'}")
    print(f"{'='*70}\n")

    # --- Load dataset ---
    print(f"Loading GPQA Diamond dataset... shard {shard}")
    samples = load_gpqa_dataset(split="train", max_samples=max_samples,shard=shard)
    print(f"  Loaded {len(samples)} samples\n")

    # --- Load model ---
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print(f"  Config: {cfg.num_hidden_layers}L, {cfg.num_key_value_heads} KV heads, head_dim={hd}")
    print(f"  Compression ratios: ", end="")
    for b in bits_list:
        print(f"{b}b={compute_compression_ratio(hd, b):.2f}x  ", end="")
    print("\n")

    # --- K-norm analysis ---
    print("K-norm analysis...")
    inp0 = tokenizer(samples[0]["question"][:200], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        o = model(**inp0, use_cache=True)
        ak = torch.cat([
            torch.norm(l[0].float().reshape(-1, l[0].shape[-1]), dim=-1)
            for l in o.past_key_values
        ])
    amp = ak.mean().item() / hd ** 0.5
    print(f"  K-norm amplification: {amp:.2f}x\n")
    del o, ak
    torch.cuda.empty_cache()

    # --- Run evaluations ---
    all_eval_results = {}

    # Baseline
    if "baseline" in modes:
        print(f"{'─'*50}")
        print(f"Running BASELINE (bf16) evaluation...")
        print(f"{'─'*50}")
        baseline_result = evaluate_gpqa(
            model, tokenizer, samples,
            mode="baseline", hd=hd, verbose=True,
        )
        all_eval_results["bf16"] = baseline_result
        print(f"  BASELINE accuracy: {baseline_result['accuracy']:.1%} "
              f"({baseline_result['correct']}/{baseline_result['total']}) "
              f"in {baseline_result['elapsed_s']:.1f}s\n")

    # TurboQuant at each bit-width
    if "turboquant" in modes:
        for bits in bits_list:
            cr = compute_compression_ratio(hd, bits)
            print(f"{'─'*50}")
            print(f"Running TURBOQUANT {bits}b ({cr:.2f}x compression) evaluation...")
            print(f"{'─'*50}")
            tq_result = evaluate_gpqa(
                model, tokenizer, samples,
                mode="turboquant", bits=bits, hd=hd, verbose=True,quant_type=quant_type
            )
            all_eval_results[f"tq_{bits}b"] = tq_result
            print(f"  TQ-{bits}b accuracy: {tq_result['accuracy']:.1%} "
                  f"({tq_result['correct']}/{tq_result['total']}) "
                  f"in {tq_result['elapsed_s']:.1f}s\n")

    # --- Print summary ---
    print(f"\n{'='*70}")
    print(f"GPQA DIAMOND RESULTS SUMMARY: {model_id}")
    print(f"{'='*70}")
    print(f"  {'Mode':<20s}  {'Accuracy':>10s}  {'Correct':>10s}  {'Time':>8s}  {'Compress':>10s}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*10}")

    baseline_acc = None
    for key, res in all_eval_results.items():
        mode_label = key
        acc_str = f"{res['accuracy']:.1%}"
        count_str = f"{res['correct']}/{res['total']}"
        time_str = f"{res['elapsed_s']:.1f}s"
        if key == "bf16":
            compress_str = "1.00x"
            baseline_acc = res["accuracy"]
        else:
            b = res["bits"]
            compress_str = f"{compute_compression_ratio(hd, b):.2f}x"
        print(f"  {mode_label:<20s}  {acc_str:>10s}  {count_str:>10s}  {time_str:>8s}  {compress_str:>10s}")

    # Show accuracy delta if both modes ran
    if baseline_acc is not None:
        print(f"\n  Accuracy delta vs baseline:")
        for key, res in all_eval_results.items():
            if key == "bf16":
                continue
            delta = res["accuracy"] - baseline_acc
            print(f"    {key}: {delta:+.1%}")

    # --- Assertions ---
    print(f"\n{'─'*50}")
    print("Running assertions...")

    assert amp < 5.0, f"K-norm amplification {amp:.1f}x unexpectedly high for Qwen3-8B"

    if "bf16" in all_eval_results:
        bl_acc = all_eval_results["bf16"]["accuracy"]
        # GPQA is hard; even bf16 might not score high, but should be above random (25%)
        assert bl_acc > 0.25, (
            f"Baseline accuracy {bl_acc:.1%} is at or below random chance"
        )

    for key, res in all_eval_results.items():
        if key == "bf16":
            continue
        tq_acc = res["accuracy"]
        # TQ should not catastrophically degrade accuracy
        # Allow up to 10 percentage points drop from baseline (or 20% if no baseline)
        if baseline_acc is not None:
            max_drop = 0.10
            assert tq_acc >= baseline_acc - max_drop, (
                f"{key} accuracy {tq_acc:.1%} dropped more than {max_drop:.0%} "
                f"from baseline {baseline_acc:.1%}"
            )
        else:
            assert tq_acc > 0.20, f"{key} accuracy {tq_acc:.1%} too low"

    print("All assertions passed!")
    print(f"{'='*70}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return all_eval_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPQA Diamond benchmark: Qwen3-8B with TurboQuant"
    )
    parser.add_argument(
        "--bits", type=float, nargs="+", default=[4],
        help="TQ bit-widths to test (e.g. 2.5 3.5 4)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max number of GPQA samples to evaluate",
    )
    parser.add_argument(
        "--mode", type=str, nargs="+", default=["baseline", "turboquant"],
        choices=["baseline", "turboquant"],
        help="Evaluation modes to run",
    )
    parser.add_argument(
        "--shard", type=int, default=None, choices=[1, 2, 3, 4],
        help="Run only the Nth quarter of the dataset (1-4). Omit to run all.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--quant_type", type=str, default=["mse"],
        choices=["prod", "mse"],
        help="Quantization type to use",
    )
    args = parser.parse_args()
    # print(extract_answer("${A}$ ${D}$"))
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")

    results = test_gpqa_qwen3_8b(
        bits_list=args.bits,
        max_samples=args.max_samples,
        modes=args.mode,
        shard=args.shard,
        quant_type=args.quant_type
    )

    if args.output:
        # Save results (strip per-sample details for readability)
        save_data = {}
        for key, res in results.items():
            save_data[key] = {
                k: v for k, v in res.items() if k != "results"
            }
            save_data[key]["per_sample"] = res["results"]
        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {args.output}")