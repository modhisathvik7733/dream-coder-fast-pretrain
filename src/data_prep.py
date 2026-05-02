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
  10% Stack-Edu Python      (some Stage 1 base coverage to preserve breadth)

Output schema (matches Dream-Coder SFT convention — see sft_dataset.py):
    {
      "input_ids":     [prompt_tokens..., response_tokens...],
      "attention_mask":[1, 1, ..., 1],
      "prompt_length": <int — # of tokens belonging to the prompt prefix>,
      "length":        <int — total tokens>
    }

train.py uses prompt_length to build a loss_mask that is 0 on the prompt+pad
positions and 1 on response positions. Only response positions are eligible
to be masked + contribute to the diffusion loss.

For raw-code samples (Stack-Edu), prompt_length = 0 — the entire sequence is
"response", matching Stage 1 base pretraining where everything is maskable.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def split_messages_to_prompt_response(messages: list) -> tuple[list, str] | None:
    """Split a multi-turn conversation into (prompt_messages, response_str).

    Last assistant turn becomes the response; everything before is the prompt
    (we'll apply the chat template with add_generation_prompt=True).
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    norm = []
    for m in messages:
        if not isinstance(m, dict):
            return None
        role = (m.get("role") or "").lower()
        if role == "human":
            role = "user"
        content = m.get("content")
        if not isinstance(content, str) or not content:
            return None
        norm.append({"role": role, "content": content})

    # Find the last assistant turn
    last_asst_idx = None
    for i in range(len(norm) - 1, -1, -1):
        if norm[i]["role"] == "assistant":
            last_asst_idx = i
            break
    if last_asst_idx is None or last_asst_idx == 0:
        return None

    prompt_messages = norm[:last_asst_idx]
    response_str = norm[last_asst_idx]["content"]
    if not response_str.strip():
        return None
    return prompt_messages, response_str


def tokenize_prompt_response(
    prompt_messages: list,
    response_str: str,
    tokenizer,
    max_seq_length: int,
) -> dict | None:
    """Tokenize prompt+response, returning the schema train.py expects.

    Mirrors Dream-Coder's `sft_dataset.py::_tokenize_static`:
      - prompt = chat_template(prompt_messages, add_generation_prompt=True)
      - response = response_str + eos
      - input_ids = prompt_ids ++ response_ids
      - attention_mask = all ones
      - prompt_length = len(prompt_ids)
    """
    try:
        prompt_str = tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        return None

    eos = tokenizer.eos_token or ""
    response_full = response_str + eos

    prompt_ids = tokenizer(
        prompt_str, add_special_tokens=False, truncation=False
    ).input_ids
    response_ids = tokenizer(
        response_full, add_special_tokens=False, truncation=False
    ).input_ids

    if len(prompt_ids) < 4 or len(response_ids) < 4:
        return None

    total = len(prompt_ids) + len(response_ids)
    if total > max_seq_length:
        # Right-truncate the response (preserve full prompt) — Dream's SFT does this implicitly
        budget = max_seq_length - len(prompt_ids)
        if budget < 8:
            return None  # prompt too long, skip
        response_ids = response_ids[:budget]
        total = len(prompt_ids) + len(response_ids)
    if total < 32:
        return None

    return {
        "input_ids": prompt_ids + response_ids,
        "attention_mask": [1] * total,
        "prompt_length": len(prompt_ids),
        "length": total,
    }


def tokenize_raw_code(text: str, tokenizer, max_seq_length: int) -> dict | None:
    """Tokenize raw code as a Stage 1 pretraining sample (no prompt prefix).

    prompt_length = 0 means the whole sequence is maskable (= base pretraining).
    """
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
        "prompt_length": 0,
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
        split = split_messages_to_prompt_response(messages)
        if split is None:
            continue
        prompt_msgs, resp = split
        item = tokenize_prompt_response(prompt_msgs, resp, tokenizer, max_seq_length)
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
        messages = row.get("messages")
        prompt_field = row.get("prompt")
        response = row.get("response") or row.get("solution") or row.get("answer")

        split = None
        if isinstance(messages, list):
            split = split_messages_to_prompt_response(messages)
        elif isinstance(prompt_field, str) and isinstance(response, str):
            fake_msgs = [
                {"role": "user", "content": prompt_field},
                {"role": "assistant", "content": response},
            ]
            split = split_messages_to_prompt_response(fake_msgs)

        if split is None:
            continue
        prompt_msgs, resp = split
        item = tokenize_prompt_response(prompt_msgs, resp, tokenizer, max_seq_length)
        if item:
            out.append(item)
        if len(out) >= target:
            break
    print(f"  Collected {len(out)} Dream-Coder-RL samples")
    return out


def collect_stack_edu_python(target: int, tokenizer, max_seq_length: int) -> list[dict]:
    """Tertiary replay: Stack-Edu Python — Stage 1 base adaptation data.

    These are raw code samples (no chat structure) — treated as base pretraining
    with prompt_length=0, so the whole sequence is maskable.
    """
    print(f"Pulling Stack-Edu Python (target {target}) — base adaptation data ...")
    out = []
    try:
        ds = load_dataset("HuggingFaceTB/stack-edu", "python", split="train", streaming=True)
    except Exception as e:
        print(f"  Stack-Edu Python load failed: {e} — trying alternative")
        try:
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
        item = tokenize_raw_code(text, tokenizer, max_seq_length)
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
    samples.extend(collect_stack_edu_python(n_stack, tokenizer, args.max_seq_length))

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
    total_response_tokens = sum(s["length"] - s["prompt_length"] for s in samples)
    print()
    print(f"Wrote {len(samples)} samples to {out_path}")
    print(f"Total tokens:    {total_tokens:,}  (~{total_tokens/1e6:.1f}M)")
    print(f"Response tokens: {total_response_tokens:,}  (~{total_response_tokens/1e6:.1f}M maskable)")
    print(f"Avg length:      {total_tokens / max(1, len(samples)):.0f} tokens")

    meta = {
        "n_samples": len(samples),
        "total_tokens": total_tokens,
        "total_response_tokens": total_response_tokens,
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
        "schema": {
            "input_ids": "list[int] — prompt tokens followed by response tokens",
            "attention_mask": "list[int] — all 1s (Dream convention; padding handled by collator)",
            "prompt_length": "int — # tokens of the prompt prefix; 0 for raw-code samples",
            "length": "int — total tokens (prompt + response)",
        },
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
