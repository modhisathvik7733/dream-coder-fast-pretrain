"""Prepare training data for low-step continued pretraining.

CRITICAL: replay data must match Dream-Coder's actual training distribution.
Using off-distribution data shifts the model and degrades original capabilities.

Dream-Coder's actual training stages (from official repo):
  - Base (adaptation): mix of Stack v2, OpenCoder, Stack-Edu, DCLM, math, etc.
  - Instruct (SFT):    inclusionAI/Ling-Coder-SFT (single dataset, 7 epochs)
  - RL:                Dream-org/Dream-Coder-RL-17k

For continued pretraining (this script's purpose), we replay from the LATEST
training stages so distribution shift is minimized:
  70% Ling-Coder-SFT        (matches Stage 2 — what the instruct model was last trained on)
  20% Dream-Coder-RL-17k    (matches Stage 3 — RL prompt distribution)
  10% Stack v2 subset       (some base coverage to preserve breadth)

This way our biased-mask training applies to data the model already knows,
purely teaching it to handle high-mask cases on familiar content.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def format_messages_to_text(messages: list, max_chars: int = 4000) -> str | None:
    """Concatenate a multi-turn conversation into a single text string.

    Filters: must end with assistant message, total length within bounds.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    if messages[-1].get("role", "").lower() != "assistant":
        return None

    parts = []
    for m in messages:
        role = m.get("role", "").lower()
        if role == "human":
            role = "user"
        content = m.get("content", "")
        if not isinstance(content, str):
            return None
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    text = "\n".join(parts)

    if len(text) < 100 or len(text) > max_chars:
        return None
    return text


def tokenize_to_sample(text: str, tokenizer, max_seq_length: int) -> dict | None:
    """Tokenize text to (input_ids, attention_mask) within seq length."""
    ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
    )
    if len(ids.input_ids) < 64:
        return None
    return {
        "input_ids": ids.input_ids,
        "attention_mask": ids.attention_mask,
        "length": len(ids.input_ids),
    }


def collect_lingcoder(target: int, tokenizer, max_seq_length: int) -> list[dict]:
    """Primary replay source: Ling-Coder-SFT (Dream-Coder-Instruct's SFT data)."""
    print(f"Pulling Ling-Coder-SFT (target {target}) — Dream-Coder's actual SFT data ...")
    out = []
    try:
        ds = load_dataset("inclusionAI/Ling-Coder-SFT", split="train", streaming=True)
    except Exception as e:
        print(f"  Ling-Coder-SFT load failed: {e}")
        return out

    for row in tqdm(ds, desc="ling-coder-sft"):
        messages = row.get("messages")
        text = format_messages_to_text(messages)
        if not text:
            continue
        item = tokenize_to_sample(text, tokenizer, max_seq_length)
        if item:
            out.append(item)
        if len(out) >= target:
            break
    print(f"  Collected {len(out)} Ling-Coder-SFT samples")
    return out


def collect_dream_rl(target: int, tokenizer, max_seq_length: int) -> list[dict]:
    """Secondary replay: Dream-Coder-RL-17k (the RL stage data)."""
    print(f"Pulling Dream-Coder-RL-17k (target {target}) — RL stage data ...")
    out = []
    try:
        ds = load_dataset("Dream-org/Dream-Coder-RL-17k", split="train", streaming=True)
    except Exception as e:
        print(f"  Dream-Coder-RL-17k load failed: {e}")
        return out

    for row in tqdm(ds, desc="dream-coder-rl"):
        # RL data may have varying schema. Try common fields.
        messages = row.get("messages") or row.get("prompt")
        response = row.get("response") or row.get("solution") or row.get("answer")

        text = None
        if messages and isinstance(messages, list):
            text = format_messages_to_text(messages)
        elif messages and isinstance(messages, str) and response:
            full = [
                {"role": "user", "content": messages},
                {"role": "assistant", "content": response if isinstance(response, str) else str(response)},
            ]
            text = format_messages_to_text(full)

        if not text:
            continue
        item = tokenize_to_sample(text, tokenizer, max_seq_length)
        if item:
            out.append(item)
        if len(out) >= target:
            break
    print(f"  Collected {len(out)} Dream-Coder-RL samples")
    return out


def collect_stack_v2_python(target: int, tokenizer, max_seq_length: int) -> list[dict]:
    """Tertiary replay: Stack v2 Python subset for breadth.

    We use the smol variant Dream-Coder used during base adaptation.
    Note: full The Stack v2 needs auth + license attestation. Using a public proxy
    or stack-edu-py which is openly available.
    """
    print(f"Pulling Stack-Edu Python (target {target}) — base adaptation data ...")
    out = []
    try:
        # Stack-Edu Python — same one Dream-Coder used in base adaptation
        ds = load_dataset("HuggingFaceTB/stack-edu", "python", split="train", streaming=True)
    except Exception as e:
        print(f"  Stack-Edu Python load failed: {e} — trying alternative")
        try:
            # Fallback: code subset of fineweb-edu equivalent
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)
        except Exception as e2:
            print(f"  Alternative also failed: {e2}")
            return out

    for row in tqdm(ds, desc="stack-edu-py"):
        text = row.get("text") or row.get("content") or row.get("code")
        if not isinstance(text, str):
            continue
        if len(text) < 200 or len(text) > 4000:
            continue
        item = tokenize_to_sample(text, tokenizer, max_seq_length)
        if item:
            out.append(item)
        if len(out) >= target:
            break
    print(f"  Collected {len(out)} Stack-Edu Python samples")
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

    # Distribution matched to Dream-Coder's training stages:
    # 70% Ling-Coder-SFT (Stage 2 — most recent)
    # 20% Dream-Coder-RL-17k (Stage 3)
    # 10% Stack-Edu Python (Stage 1 base coverage)
    n_ling = int(args.target_samples * 0.70)
    n_rl = int(args.target_samples * 0.20)
    n_stack = args.target_samples - n_ling - n_rl

    samples = []
    samples.extend(collect_lingcoder(n_ling, tokenizer, args.max_seq_length))
    samples.extend(collect_dream_rl(n_rl, tokenizer, args.max_seq_length))
    samples.extend(collect_stack_v2_python(n_stack, tokenizer, args.max_seq_length))

    if not samples:
        print("ERROR: no samples collected from any source.")
        print("Check internet connectivity and dataset access (some require auth).")
        exit(1)

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

    meta = {
        "n_samples": len(samples),
        "total_tokens": total_tokens,
        "max_seq_length": args.max_seq_length,
        "avg_length": total_tokens / max(1, len(samples)),
        "data_mix": {
            "ling-coder-sft": 0.70,
            "dream-coder-rl-17k": 0.20,
            "stack-edu-python": 0.10,
        },
        "rationale": (
            "Matches Dream-Coder's actual training distribution (Stage 2 SFT + "
            "Stage 3 RL + Stage 1 base coverage). Using off-distribution data "
            "would shift the model and degrade original capabilities."
        ),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
