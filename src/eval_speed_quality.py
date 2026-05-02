"""Evaluate fast Dream-Coder vs vanilla on HumanEval+ across multiple step counts.

Reports:
  - Pass@1 at each step count (4, 8, 16, 32 steps per block)
  - Wall time per problem
  - Comparison vs vanilla baseline at 32 steps
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

CODE_RE_FULL = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
CODE_RE_OPEN = re.compile(r"```(?:python|py)?\s*\n(.*)$", re.DOTALL)


def extract_code(text: str) -> str:
    m = CODE_RE_FULL.search(text)
    if m:
        return m.group(1).strip()
    m = CODE_RE_OPEN.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def run_test(code: str, test_code: str, entry_point: str, timeout: int = 10) -> bool:
    full = (
        code + "\n\n" + test_code
        + f"\n\ntry:\n    check({entry_point})\n    print('__OK__')\nexcept: pass\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        p = subprocess.run(["python", path], capture_output=True, text=True, timeout=timeout)
        return "__OK__" in p.stdout
    except subprocess.TimeoutExpired:
        return False


def evaluate_at_steps(
    model, tokenizer, problems, steps_per_block: int, max_new_tokens: int = 256, limit: int = 50
):
    """Run model on `limit` problems at given step count. Returns (pass_rate, avg_time)."""
    n_pass = 0
    times = []
    for i, (task_id, row) in enumerate(problems[:limit]):
        prompt = (
            f"<|im_start|>user\n"
            f"Complete the following Python function. Return only the full function "
            f"definition in a ```python code block.\n\n"
            f"```python\n{row['prompt']}\n```\n"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to("cuda")

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.diffusion_generate(
                inp.input_ids,
                attention_mask=inp.attention_mask,
                max_new_tokens=max_new_tokens,
                steps=steps_per_block * (max_new_tokens // 32),  # approx
                temperature=0.2,
                top_p=0.95,
                alg="entropy",
            )
        torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)

        seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
        completion = tokenizer.decode(seq[inp.input_ids.shape[1]:], skip_special_tokens=True)
        code = extract_code(completion)

        if run_test(code, row["test"], row["entry_point"]):
            n_pass += 1

        if (i + 1) % 10 == 0:
            print(f"  steps={steps_per_block:3d}  [{i+1}/{limit}]  "
                  f"pass@1={n_pass/(i+1):.3f}  avg_time={sum(times)/len(times):.2f}s")

    return n_pass / limit, sum(times) / len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Trained checkpoint to evaluate")
    ap.add_argument("--baseline_path", required=False, help="Vanilla Dream-Coder for comparison")
    ap.add_argument("--steps_per_block", nargs="+", type=int, default=[4, 8, 16, 32])
    ap.add_argument("--benchmark", default="humaneval_plus")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Load benchmark
    if args.benchmark == "humaneval_plus":
        from evalplus.data import get_human_eval_plus
        problems = list(get_human_eval_plus().items())
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    results = {"model_path": args.model_path, "limit": args.limit, "results": {}}

    # === Eval the trained model ===
    print(f"Loading trained model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()

    print()
    for steps in args.steps_per_block:
        print(f"=== Trained model at steps_per_block={steps} ===")
        rate, avg_t = evaluate_at_steps(model, tokenizer, problems, steps, limit=args.limit)
        results["results"][f"trained_steps_{steps}"] = {
            "pass_at_1": rate,
            "avg_time_seconds": avg_t,
        }
        print(f"  pass@1: {rate:.4f}  avg_time: {avg_t:.2f}s")
        print()

    # Free memory before loading baseline
    del model
    torch.cuda.empty_cache()

    # === Eval vanilla baseline (optional) ===
    if args.baseline_path:
        print(f"Loading baseline from {args.baseline_path}")
        bmodel = AutoModel.from_pretrained(
            args.baseline_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda").eval()
        btok = AutoTokenizer.from_pretrained(args.baseline_path, trust_remote_code=True)

        print()
        print("=== Vanilla baseline at steps_per_block=32 ===")
        rate, avg_t = evaluate_at_steps(bmodel, btok, problems, 32, limit=args.limit)
        results["results"]["baseline_steps_32"] = {
            "pass_at_1": rate,
            "avg_time_seconds": avg_t,
        }
        print(f"  pass@1: {rate:.4f}  avg_time: {avg_t:.2f}s")

    # === Save ===
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.output}")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"{'config':<25} {'pass@1':>10} {'time(s)':>10}")
    print("-" * 50)
    for key, val in results["results"].items():
        print(f"{key:<25} {val['pass_at_1']:>10.4f} {val['avg_time_seconds']:>10.2f}")


if __name__ == "__main__":
    main()
