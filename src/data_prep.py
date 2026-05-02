"""Prepare training data for low-step continued pretraining.

Mix:
  40% Python competitive code (anti-forgetting from KodCode-V1)
  30% reasoning traces (preserves R1 distillation in Dream-Coder)
  30% general code corpus (extends competence)

Output: a single JSONL with {input_ids: [...], labels: [...]} ready for SFT.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def filter_python(row, min_chars: int = 50, max_chars: int = 4000) -> bool:
    """Quality filter for Python code training rows."""
    sol = row.get("solution") or row.get("code") or row.get("output") or ""
    if not isinstance(sol, str):
        return False
    return min_chars <= len(sol) <= max_chars and "def " in sol


def format_instruction_pair(prompt: str, response: str, tokenizer, max_seq_length: int) -> dict | None:
    """Tokenize a single (prompt, response) pair into ids + labels."""
    full = f"{prompt.strip()}\n\n{response.strip()}"
    ids = tokenizer(full, add_special_tokens=True, truncation=True, max_length=max_seq_length)
    if len(ids.input_ids) < 64:
        return None
    return {
        "input_ids": ids.input_ids,
        "attention_mask": ids.attention_mask,
        "length": len(ids.input_ids),
    }


def collect_python_replay(target: int, tokenizer, max_seq_length: int) -> list[dict]:
    """40% of mix: Python competitive code from KodCode-V1."""
    print(f"Pulling Python replay data (target {target}) ...")
    out = []
    try:
        ds = load_dataset("KodCode/KodCode-V1", split="train", streaming=True)
    except Exception as e:
        print(f"  KodCode load failed: {e}; skipping.")
        return []

    for row in tqdm(ds, desc="kodcode"):
        if not filter_python(row):
            continue
        prompt = row.get("question") or row.get("problem") or ""
        sol = row.get("solution") or row.get("code") or ""
        if not (prompt and sol):
            continue
        item = format_instruction_pair(prompt, sol, tokenizer, max_seq_length)
        if item is not None:
            out.append(item)
        if len(out) >= target:
            break
    print(f"  Got {len(out)} Python samples")
    return out


def collect_reasoning(target: int, tokenizer, max_seq_length: int) -> list[dict]:
    """30% of mix: reasoning traces (preserves R1 distillation)."""
    print(f"Pulling reasoning data (target {target}) ...")
    out = []
    try:
        ds = load_dataset("nvidia/OpenCodeReasoning", split="train", streaming=True)
    except Exception as e:
        print(f"  OpenCodeReasoning load failed: {e}; skipping.")
        return []

    for row in tqdm(ds, desc="opencodereasoning"):
        prompt = row.get("question") or row.get("input") or ""
        response = row.get("response") or row.get("output") or ""
        if not (isinstance(prompt, str) and isinstance(response, str)):
            continue
        if "<think>" not in response or "</think>" not in response:
            continue
        if len(response) < 200 or len(response) > 8000:
            continue
        item = format_instruction_pair(prompt, response, tokenizer, max_seq_length)
        if item is not None:
            out.append(item)
        if len(out) >= target:
            break
    print(f"  Got {len(out)} reasoning samples")
    return out


def collect_general_code(target: int, tokenizer, max_seq_length: int) -> list[dict]:
    """30% of mix: general code corpus (Bespoke-Stratos as fallback)."""
    print(f"Pulling general code data (target {target}) ...")
    out = []
    try:
        ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train", streaming=True)
    except Exception as e:
        print(f"  Bespoke-Stratos load failed: {e}; skipping.")
        return []

    for row in tqdm(ds, desc="bespoke-stratos"):
        prompt = row.get("system") or row.get("conversations", [{}])[0].get("value", "") if isinstance(row.get("conversations"), list) else ""
        response = ""
        convs = row.get("conversations")
        if isinstance(convs, list) and len(convs) >= 2:
            prompt = convs[0].get("value", "") if isinstance(convs[0], dict) else ""
            response = convs[1].get("value", "") if isinstance(convs[1], dict) else ""

        if not (prompt and response):
            continue
        if len(response) < 100 or len(response) > 6000:
            continue
        item = format_instruction_pair(prompt, response, tokenizer, max_seq_length)
        if item is not None:
            out.append(item)
        if len(out) >= target:
            break
    print(f"  Got {len(out)} general code samples")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--target_samples", type=int, default=50000)
    ap.add_argument("--max_seq_length", type=int, default=768)
    ap.add_argument("--tokenizer_path", type=str,
                    default="/workspace/models/dream-coder-7b-instruct")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # 40/30/30 split
    n_python = int(args.target_samples * 0.40)
    n_reason = int(args.target_samples * 0.30)
    n_general = args.target_samples - n_python - n_reason

    samples = []
    samples.extend(collect_python_replay(n_python, tokenizer, args.max_seq_length))
    samples.extend(collect_reasoning(n_reason, tokenizer, args.max_seq_length))
    samples.extend(collect_general_code(n_general, tokenizer, args.max_seq_length))

    random.seed(42)
    random.shuffle(samples)

    out_path = args.output_dir / "train.jsonl"
    with out_path.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    total_tokens = sum(s["length"] for s in samples)
    print()
    print(f"Wrote {len(samples)} samples to {out_path}")
    print(f"Total tokens: {total_tokens:,}  (~{total_tokens/1e6:.1f}M)")
    print(f"Avg length: {total_tokens / max(1, len(samples)):.0f} tokens")

    # Also write a meta file
    meta = {
        "n_samples": len(samples),
        "total_tokens": total_tokens,
        "max_seq_length": args.max_seq_length,
        "avg_length": total_tokens / max(1, len(samples)),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
