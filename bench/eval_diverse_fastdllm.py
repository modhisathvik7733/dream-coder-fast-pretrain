"""Dream-Coder diverse best-of-8 — fast_dllm + (optional) torch.compile.

Adapted from `eval_amplified_diverse_dreamcoder.py` (the Dream-Coder port of
DiffuCoder's diverse best-of-8 eval). Adds fast_dllm KV-cache on top.

Stacks 3 speedup techniques:
  1. fast_dllm KV-cache (dual_cache=True)              — ~2× per attempt (proven)
  2. torch.compile(mode="reduce-overhead")             — ~1.5-2× per attempt (when stable)
  3. Diverse best-of-8 with first-pass early-stop      — accuracy via attempt budget

Per-attempt cost is multiplicative on top of avg_attempts (which is typically
1.3-1.8 with diverse best-of-8). Combined: ~3-5× faster wall-clock vs vanilla
AutoModel diverse best-of-8 at the same pass@1.

Usage (after bench/00_setup_fast_dllm.sh has run on a vast.ai box):

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
        # vast.ai images often have python3 but no `python` symlink.
        p = subprocess.run(["python3", path], capture_output=True, text=True, timeout=timeout)
        return "__OK__" in p.stdout
    except subprocess.TimeoutExpired:
        return False


# 8 diverse configs. We use `steps_per_block` (not raw `steps`) so the configs
# work for any max_new_tokens that divides BLOCK_LENGTH evenly. fast_dllm's
# block scheduler requires `total_steps % num_blocks == 0`; computing total from
# steps_per_block × num_blocks guarantees that automatically.
#
#   steps_per_block=32 → "full quality" pass (was steps=256 with 8 blocks)
#   steps_per_block=16 → "fast" pass         (was steps=128 with 8 blocks)
DIVERSE_CONFIGS = [
    {"alg": "entropy",      "alg_temp": 0.0, "temperature": 0.2, "top_p": 0.95, "steps_per_block": 32},
    {"alg": "entropy",      "alg_temp": 0.0, "temperature": 0.5, "top_p": 0.95, "steps_per_block": 32},
    {"alg": "entropy",      "alg_temp": 0.3, "temperature": 0.7, "top_p": 0.95, "steps_per_block": 16},
    {"alg": "entropy",      "alg_temp": 0.5, "temperature": 0.9, "top_p": 0.92, "steps_per_block": 16},
    {"alg": "maskgit_plus", "alg_temp": 0.0, "temperature": 0.4, "top_p": 0.95, "steps_per_block": 32},
    {"alg": "maskgit_plus", "alg_temp": 0.3, "temperature": 0.7, "top_p": 0.95, "steps_per_block": 16},
    {"alg": "topk_margin",  "alg_temp": 0.0, "temperature": 0.4, "top_p": 0.95, "steps_per_block": 32},
    {"alg": "topk_margin",  "alg_temp": 0.3, "temperature": 0.7, "top_p": 0.95, "steps_per_block": 16},
]


# Constants for fast_dllm's block scheduler.
BLOCK_LENGTH = 32
THRESHOLD = 0.9


def setup_fast_dllm_model(dream_repo: Path, model_path: str, use_compile: bool, quant: str):
    """Load model via fast_dllm DreamModel + optional quantization + monkey-patch + warmup."""
    sys.path.insert(0, str(dream_repo / "instruct"))
    from src.inference.fast_dllm.modeling_dream import DreamModel  # noqa: E402
    from src.inference.fast_dllm.generation_utils_block import (  # noqa: E402
        DreamGenerationMixin,
    )

    print(f"Loading {model_path} via fast_dllm DreamModel "
          f"(quant={quant or 'bf16'}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    load_kwargs: dict = {"torch_dtype": torch.bfloat16}
    needs_to_cuda = True

    if quant == "int8":
        from transformers import BitsAndBytesConfig  # noqa: E402
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        # bitsandbytes places weights on cuda automatically — don't .to() afterward
        needs_to_cuda = False
        load_kwargs.pop("torch_dtype")  # bnb manages dtype
    elif quant == "int4":
        from transformers import BitsAndBytesConfig  # noqa: E402
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        needs_to_cuda = False
        load_kwargs.pop("torch_dtype")

    model = DreamModel.from_pretrained(model_path, **load_kwargs)
    if needs_to_cuda:
        model = model.to("cuda")
    model = model.eval()
    print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Bind block-aware diffusion_generate (the path that supports dual_cache).
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

    # Warmup with a representative config so the cache + (optional) compile graph are hot.
    # NOTE: steps must be a multiple of num_blocks = max_new_tokens / block_length.
    print("Warming up ...")
    warmup_messages = [{"role": "user", "content": "Write add(a,b)."}]
    warmup_inp = tokenizer.apply_chat_template(
        warmup_messages, return_tensors="pt", return_dict=True,
        add_generation_prompt=True,
    )
    warmup_max_new = 256  # 8 blocks of 32
    warmup_steps = 16 * (warmup_max_new // BLOCK_LENGTH)  # 16 steps/block × 8 = 128
    t0 = time.time()
    with torch.no_grad():
        _ = model.diffusion_generate(
            warmup_inp.input_ids.to("cuda"),
            attention_mask=warmup_inp.attention_mask.to("cuda"),
            max_new_tokens=warmup_max_new, steps=warmup_steps,
            temperature=0.2, top_p=0.95, alg="entropy", alg_temp=0.0,
            dual_cache=True, block_length=BLOCK_LENGTH, threshold=THRESHOLD,
        )
    print(f"Warmup time: {time.time() - t0:.1f}s")

    return tokenizer, model


def evaluate(tokenizer, model, items, max_new_tokens: int):
    # Precompute total steps from each config's steps_per_block. fast_dllm requires
    # steps % num_blocks == 0 — guaranteed when steps = steps_per_block × num_blocks.
    assert max_new_tokens % BLOCK_LENGTH == 0, (
        f"max_new_tokens ({max_new_tokens}) must be a multiple of "
        f"BLOCK_LENGTH ({BLOCK_LENGTH})"
    )
    num_blocks = max_new_tokens // BLOCK_LENGTH
    resolved_configs = []
    for cfg in DIVERSE_CONFIGS:
        resolved = {k: v for k, v in cfg.items() if k != "steps_per_block"}
        resolved["steps"] = cfg["steps_per_block"] * num_blocks
        resolved_configs.append(resolved)

    results = []
    n_pass = 0
    attempts_used = []
    t_start = time.time()

    for i, (task_id, row) in enumerate(items):
        # Use Dream-Coder's chat template (matches the published model card).
        messages = [{
            "role": "user",
            "content": (
                f"Complete the following Python function. Return only the full function "
                f"definition in a ```python code block.\n\n"
                f"```python\n{row['prompt']}\n```"
            ),
        }]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        found = False
        used = 0

        for j, cfg in enumerate(resolved_configs):
            used = j + 1
            with torch.no_grad():
                out = model.diffusion_generate(
                    input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    dual_cache=True, block_length=BLOCK_LENGTH, threshold=THRESHOLD,
                    **cfg,
                )
            seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
            code = extract_code(
                tokenizer.decode(seq[input_ids.shape[1]:], skip_special_tokens=True)
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
    ap.add_argument("--max_new_tokens", type=int, default=256,
                    help="Lower = faster (linear). 192 typically still passes most "
                         "HumanEval+ problems; 256 covers all. Try 192 for a free 25%% speedup.")
    ap.add_argument("--quant", type=str, default="",
                    choices=["", "int8", "int4"],
                    help="Quantization: '' (bf16, default), 'int8' (~1.5-2x faster, "
                         "~-2 pts pass@1), 'int4' (~2-3x faster, ~-4 pts pass@1). "
                         "Requires bitsandbytes installed.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Evaluate only the first N HumanEval+ problems (0 = all 164).")
    ap.add_argument("--no-compile", dest="use_compile", action="store_false", default=True,
                    help="Disable torch.compile (use if compile errors).")
    ap.add_argument("--out", type=Path,
                    default=Path("/workspace/results/dreamcoder_diverse_fastdllm.json"))
    args = ap.parse_args()

    tokenizer, model = setup_fast_dllm_model(
        args.dream_repo, args.model_path, args.use_compile, args.quant
    )

    items = list(get_human_eval_plus().items())
    if args.limit:
        items = items[: args.limit]

    print(
        f"\nEvaluating on {len(items)} problems "
        f"(max_new_tokens={args.max_new_tokens}, "
        f"compile={args.use_compile}, fast_dllm dual_cache=True) ...\n"
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
    techniques = ["fast_dllm dual_cache"]
    if args.use_compile:
        techniques.append("torch.compile")
    if args.quant:
        techniques.append(args.quant)
    print(f"  speedup techniques: {' + '.join(techniques)}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
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
                "block_length": BLOCK_LENGTH,
                "threshold": THRESHOLD,
                "quantization": args.quant or "bf16",
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
