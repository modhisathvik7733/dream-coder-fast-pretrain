"""Benchmark vanilla Dream-Coder vs fast_dllm KV-cache (`dual_cache=True`).

Goal: confirm the published 3-5× speedup at FULL quality without any training.

Compares 3 configs on the same prompts:
  1. Vanilla AutoModel.diffusion_generate (no cache)            ← baseline
  2. fast_dllm DreamModel + dual_cache=True                     ← the magic
  3. (Optional) fast_dllm + threshold-based early stopping      ← stacked

Prints per-config wall time and prints generated code so you can eyeball
that quality is preserved.

Usage:
    python fast_dllm_smoke.py \\
        --dream_repo /workspace/Dream-Coder \\
        --model_path /workspace/models/dream-coder-7b-instruct \\
        --max_new_tokens 256 \\
        --steps 256
"""
from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


PROMPTS = [
    # 1. Easy — Fibonacci
    "Write a Python function `fib(n)` that returns the n-th Fibonacci number "
    "using memoization. Return only the function definition in a python code block.",

    # 2. Medium — quicksort
    "Write a Python function `quicksort(arr)` that sorts a list in-place using "
    "the Lomuto partition scheme. Return only the function definition in a python "
    "code block.",

    # 3. Slightly harder — LRU cache
    "Implement a Python class `LRUCache` with `get(key)` and `put(key, value)` methods, "
    "both O(1). Use a dict + doubly-linked list. Return only the class definition in "
    "a python code block.",
]


def build_chat_prompt(user_text: str, tokenizer) -> str:
    """Apply Dream-Coder's chat template to a single-turn user prompt."""
    msgs = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False
    )


def time_generation(
    model, tokenizer, prompt_text: str, *, steps: int, max_new_tokens: int,
    extra_kwargs: dict, label: str,
) -> tuple[float, str]:
    """Run one generation, return (seconds, decoded_text)."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.diffusion_generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            steps=steps,
            temperature=0.2,
            top_p=0.95,
            alg="entropy",
            output_history=False,
            return_dict_in_generate=True,
            **extra_kwargs,
        )
    torch.cuda.synchronize()
    dt = time.time() - t0

    seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
    completion = tokenizer.decode(seq[inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return dt, completion


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dream_repo", type=Path, required=True,
                    help="Path to a clone of github.com/DreamLM/Dream-Coder. "
                         "Used to import src.inference.fast_dllm.*")
    ap.add_argument("--model_path", type=str, required=True,
                    help="Path to Dream-Coder-v0-Instruct-7B (local dir or HF repo id)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--steps", type=int, default=256,
                    help="Total denoising steps. Default 256 = 32 steps/block × 8 blocks.")
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="Confidence threshold for early stopping (fast_dllm only).")
    ap.add_argument("--skip_baseline", action="store_true",
                    help="Skip vanilla baseline (faster smoke).")
    args = ap.parse_args()

    # Make `src.inference.fast_dllm.*` importable
    sys.path.insert(0, str(args.dream_repo / "instruct"))

    print("=" * 72)
    print("Dream-Coder fast_dllm smoke benchmark")
    print(f"  Model:          {args.model_path}")
    print(f"  Dream-Coder src:{args.dream_repo}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  steps (total):  {args.steps}  (= {args.steps // (args.max_new_tokens // args.block_length)} per block)")
    print(f"  block_length:   {args.block_length}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Tokenizer (shared)
    # ------------------------------------------------------------------
    print("\n[setup] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # ------------------------------------------------------------------
    # Run 1 — vanilla AutoModel.diffusion_generate (no cache)
    # ------------------------------------------------------------------
    if not args.skip_baseline:
        print("\n[run 1/2] Vanilla AutoModel.diffusion_generate (no cache)")
        print("           Loading model ...")
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda").eval()

        print("           Warming up ...")
        time_generation(
            model, tokenizer, build_chat_prompt(PROMPTS[0], tokenizer),
            steps=args.steps, max_new_tokens=args.max_new_tokens,
            extra_kwargs={}, label="warmup",
        )

        vanilla_times = []
        for i, p in enumerate(PROMPTS):
            chat_prompt = build_chat_prompt(p, tokenizer)
            dt, out_text = time_generation(
                model, tokenizer, chat_prompt,
                steps=args.steps, max_new_tokens=args.max_new_tokens,
                extra_kwargs={}, label=f"vanilla[{i}]",
            )
            vanilla_times.append(dt)
            print(f"  prompt {i}: {dt:6.2f} s")
            print(f"  ---- generated ----")
            print(out_text[:400])
            print(f"  -------------------")

        del model
        torch.cuda.empty_cache()
    else:
        vanilla_times = None
        print("\n[run 1/2] SKIPPED (--skip_baseline)")

    # ------------------------------------------------------------------
    # Run 2 — fast_dllm DreamModel + dual_cache=True
    # ------------------------------------------------------------------
    print("\n[run 2/2] fast_dllm DreamModel + dual_cache=True")
    print("           Loading model + monkey-patching diffusion_generate ...")

    from src.inference.fast_dllm.modeling_dream import DreamModel  # noqa: E402
    from src.inference.fast_dllm.generation_utils_block import (  # noqa: E402
        DreamGenerationMixin,
    )

    fast_model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda").eval()

    # Bind the block-aware diffusion_generate from generation_utils_block
    fast_model.diffusion_generate = types.MethodType(
        DreamGenerationMixin.diffusion_generate, fast_model
    )
    fast_model._sample = types.MethodType(
        DreamGenerationMixin._sample, fast_model
    )

    print("           Warming up ...")
    time_generation(
        fast_model, tokenizer, build_chat_prompt(PROMPTS[0], tokenizer),
        steps=args.steps, max_new_tokens=args.max_new_tokens,
        extra_kwargs={
            "dual_cache": True,
            "block_length": args.block_length,
            "threshold": args.threshold,
        },
        label="warmup",
    )

    fast_times = []
    for i, p in enumerate(PROMPTS):
        chat_prompt = build_chat_prompt(p, tokenizer)
        dt, out_text = time_generation(
            fast_model, tokenizer, chat_prompt,
            steps=args.steps, max_new_tokens=args.max_new_tokens,
            extra_kwargs={
                "dual_cache": True,
                "block_length": args.block_length,
                "threshold": args.threshold,
            },
            label=f"fast[{i}]",
        )
        fast_times.append(dt)
        print(f"  prompt {i}: {dt:6.2f} s")
        print(f"  ---- generated ----")
        print(out_text[:400])
        print(f"  -------------------")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if vanilla_times is not None:
        v_mean = sum(vanilla_times) / len(vanilla_times)
        f_mean = sum(fast_times) / len(fast_times)
        speedup = v_mean / f_mean if f_mean > 0 else 0.0

        print(f"{'prompt':<10} {'vanilla (s)':>14} {'fast (s)':>14} {'speedup':>10}")
        print("-" * 52)
        for i, (v, f) in enumerate(zip(vanilla_times, fast_times)):
            sp = v / f if f > 0 else 0.0
            print(f"{i:<10} {v:>14.2f} {f:>14.2f} {sp:>9.2f}×")
        print("-" * 52)
        print(f"{'mean':<10} {v_mean:>14.2f} {f_mean:>14.2f} {speedup:>9.2f}×")
    else:
        f_mean = sum(fast_times) / len(fast_times)
        print(f"fast mean: {f_mean:.2f} s/prompt (no baseline to compare)")

    print("\nIf speedup > 2× and the generated code looks fine, fast_dllm is a")
    print("free win — drop it into your inference path with no training cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
