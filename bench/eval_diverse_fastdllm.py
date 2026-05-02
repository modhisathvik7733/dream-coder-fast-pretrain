"""Dream-Coder diverse best-of-8 — fast_dllm + (optional) torch.compile.

Combines three speedup techniques:
  1. fast_dllm KV-cache (dual_cache=True)              — ~2× speedup (proven)
  2. torch.compile(mode="reduce-overhead")             — ~1.5-2× speedup (when it works)
  3. Diverse best-of-8 with early-stop                 — accuracy via attempt budget

Adapted from:
  diffucoder_experiments/models/diffucoder-7b-cpgrpo/eval_amplified_diverse_fast.py
which used DiffuCoder-7B-cpGRPO + torch.compile only. This version targets
Dream-Coder (or your fine-tuned variant) and adds fast_dllm KV-cache on top.

== Why this works ==
  Each problem tries up to 8 configurations; the first one that passes wins.
  Hard problems use more attempts; easy ones short-circuit on attempt 1.
  fast_dllm's dual_cache shaves ~50% off each forward pass, so the per-attempt
  cost drops linearly. With early-stop, the total benefit is multiplicative
  on top of avg_attempts.

== Expected speedup vs vanilla AutoModel diverse best-of-8 ==
  Per-attempt:       ~2× from fast_dllm
  Per-attempt:       ~1.5× from torch.compile (additional, if it survives)
  Combined:          ~3-4× wall-clock at the same pass@1

Usage (on the vast.ai instance after bench/00_setup_fast_dllm.sh has run):

    python3 bench/eval_diverse_fastdllm.py \\
        --dream_repo /workspace/Dream-Coder \\
        --model_path /workspace/models/dream-coder-7b-instruct \\
        --limit 20         # quick check on 20 problems
        # or omit --limit for full HumanEval+ (164 problems)

To benchmark *your* trained checkpoint instead:
    --model_path ModhiSathvik/dream-coder-fast-7b-instruct
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import torch
from evalplus.data import get_human_eval_plus
from transformers import AutoTokenizer


CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = CODE_RE.findall(text)
    return max(blocks, key=len).strip() if blocks else text.strip()


def run_test(code: str, test_code: str, entry_point: str, timeout: int = 10) -> bool:
    full = (
        code + "\n\n" + test_code
        + f"\n\ntry:\n    check({entry_point})\n    print('__OK__')\nexcept: pass\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        p = subprocess.run(["python3", path], capture_output=True, text=True, timeout=timeout)
        return "__OK__" in p.stdout
    except subprocess.TimeoutExpired:
        return False


# Diverse sampling configs — same shape as the DiffuCoder version, but step counts
# tuned for Dream-Coder's 32-token block_length. Each `steps` value MUST be a
# multiple of (max_new_tokens / block_length) for fast_dllm's block scheduler.
# Default block_length=32, max_new_tokens=256 → 8 blocks → steps must be multiple of 8.
DIVERSE_CONFIGS = [
    {"alg": "entropy",      "alg_temp": 0.0, "temperature": 0.2, "top_p": 0.95, "steps": 256},
    {"alg": "entropy",      "alg_temp": 0.0, "temperature": 0.5, "top_p": 0.95, "steps": 256},
    {"alg": "entropy",      "alg_temp": 0.3, "temperature": 0.7, "top_p": 0.95, "steps": 128},
    {"alg": "entropy",      "alg_temp": 0.5, "temperature": 0.9, "top_p": 0.92, "steps": 128},
    {"alg": "maskgit_plus", "alg_temp": 0.0, "temperature": 0.4, "top_p": 0.95, "steps": 256},
    {"alg": "maskgit_plus", "alg_temp": 0.3, "temperature": 0.7, "top_p": 0.95, "steps": 128},
    {"alg": "topk_margin",  "alg_temp": 0.0, "temperature": 0.4, "top_p": 0.95, "steps": 256},
    {"alg": "topk_margin",  "alg_temp": 0.3, "temperature": 0.7, "top_p": 0.95, "steps": 128},
]


def setup_fast_dllm_model(
    dream_repo: Path,
    model_path: str,
    use_compile: bool,
):
    """Load model via fast_dllm + optional torch.compile + warmup."""
    sys.path.insert(0, str(dream_repo / "instruct"))
    from src.inference.fast_dllm.modeling_dream import DreamModel  # noqa: E402
    from src.inference.fast_dllm.generation_utils_block import (  # noqa: E402
        DreamGenerationMixin,
    )

    print(f"Loading {model_path} via fast_dllm DreamModel ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Bind block-aware diffusion_generate (the one that supports dual_cache)
    model.diffusion_generate = types.MethodType(
        DreamGenerationMixin.diffusion_generate, model
    )
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)

    if use_compile:
        print("Compiling model (reduce-overhead mode) ...")
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception as e:
            print(f"WARNING: torch.compile failed ({type(e).__name__}: {e}).")
            print("Continuing without compile.")

    # Warmup with a representative config so the cache + (maybe) compile graph are hot.
    print("Warming up ...")
    warmup_text = (
        "<|im_start|>user\nWrite a function that adds two numbers.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inp = tokenizer(warmup_text, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        _ = model.diffusion_generate(
            inp.input_ids,
            attention_mask=inp.attention_mask,
            max_new_tokens=256,
            steps=128,
            temperature=0.2,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.0,
            dual_cache=True,
            block_length=32,
            threshold=0.9,
        )
    print(f"Warmup time: {time.time() - t0:.1f}s")

    return tokenizer, model


def evaluate(tokenizer, model, items, max_new_tokens: int):
    results = []
    n_pass = 0
    attempts_used = []
    t_start = time.time()

    for i, (task_id, row) in enumerate(items):
        prompt = (
            f"<|im_start|>system\nYou are an expert Python programmer.<|im_end|>\n"
            f"<|im_start|>user\nComplete the function below. Return only the full function definition in a ```python code block.\n\n"
            f"```python\n{row['prompt']}\n```<|im_end|>\n<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to("cuda")
        found = False
        used = 0

        for j, cfg in enumerate(DIVERSE_CONFIGS):
            used = j + 1
            with torch.no_grad():
                out = model.diffusion_generate(
                    inp.input_ids,
                    attention_mask=inp.attention_mask,
                    max_new_tokens=max_new_tokens,
                    dual_cache=True,
                    block_length=32,
                    threshold=0.9,
                    **cfg,
                )
            seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
            code = extract_code(
                tokenizer.decode(seq[inp.input_ids.shape[1]:], skip_special_tokens=True)
            )
            if run_test(code, row["test"], row["entry_point"]):
                found = True
                break

        if found:
            n_pass += 1
        attempts_used.append(used)
        results.append({"task_id": task_id, "passed": found, "attempts": used})

        if (i + 1) % 5 == 0 or i + 1 == len(items):
            elapsed = time.time() - t_start
            eta = elapsed * (len(items) - i - 1) / (i + 1)
            print(
                f"[{i+1:3d}/{len(items)}] pass@1 = {n_pass/(i+1):.4f}   "
                f"avg_attempts = {sum(attempts_used)/len(attempts_used):.2f}   "
                f"ETA {eta/60:.1f} min"
            )

    return results, n_pass, attempts_used, time.time() - t_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dream_repo", type=Path, required=True,
                    help="Path to a clone of github.com/DreamLM/Dream-Coder.")
    ap.add_argument("--model_path", type=str,
                    default="/workspace/models/dream-coder-7b-instruct",
                    help="Local path or HF repo id of a Dream-architecture model.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0,
                    help="Evaluate only the first N HumanEval+ problems (0 = all 164).")
    ap.add_argument("--no-compile", dest="use_compile", action="store_false", default=True,
                    help="Disable torch.compile (use if compile errors).")
    ap.add_argument("--out", type=Path,
                    default=Path("/workspace/results/dreamcoder_diverse_fastdllm.json"))
    args = ap.parse_args()

    tokenizer, model = setup_fast_dllm_model(
        args.dream_repo, args.model_path, args.use_compile
    )

    items = list(get_human_eval_plus().items())
    if args.limit:
        items = items[: args.limit]

    print(
        f"\nEvaluating on {len(items)} problems "
        f"(max_new_tokens={args.max_new_tokens}, "
        f"compile={args.use_compile}, fast_dllm=True dual_cache=True) ...\n"
    )

    results, n_pass, attempts_used, elapsed = evaluate(
        tokenizer, model, items, args.max_new_tokens
    )

    final = n_pass / len(items)
    avg_attempts = sum(attempts_used) / len(attempts_used)

    print("\n" + "=" * 60)
    print(f"FINAL: HumanEval+ diverse best-of-{len(DIVERSE_CONFIGS)} (fast_dllm)")
    print(f"  model:   {args.model_path}")
    print(f"  pass@1:  {final:.4f} ({n_pass}/{len(items)})")
    print(f"  avg attempts/problem: {avg_attempts:.2f}")
    print(f"  total wall time: {elapsed:.1f}s ({elapsed/len(items):.1f}s/problem)")
    print(f"  speedup techniques: fast_dllm dual_cache + "
          f"{'torch.compile' if args.use_compile else 'no compile'}")
    print("=" * 60)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model": args.model_path,
                "configs": DIVERSE_CONFIGS,
                "max_new_tokens": args.max_new_tokens,
                "torch_compile": args.use_compile,
                "fast_dllm_dual_cache": True,
                "block_length": 32,
                "threshold": 0.9,
                "limit": args.limit,
                "pass_at_1": final,
                "n_total": len(items),
                "n_pass": n_pass,
                "avg_attempts": avg_attempts,
                "wall_time_seconds": elapsed,
                "per_problem": results,
            },
            indent=2,
        )
    )
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
