"""Continued pretraining of Dream-Coder for low-step block decoding.

Mechanism (masked diffusion training with biased mask ratio):
  Standard masked-diffusion training samples mask ratios uniformly in [0, 1].
  Lower ratios = late denoising (few tokens still masked, easy task).
  Higher ratios = early denoising (most tokens masked, hard task — needs big jumps).

  To teach the model to make BIGGER jumps per step (so 4 steps suffice instead of 32),
  we bias the mask ratio toward HIGH values. The model trains predominantly on
  hard high-mask-ratio cases, learning to commit many tokens at once confidently.

Loss formulation:
  Standard cross-entropy on masked positions only (HF Trainer ignore_index=-100).
  Logits are shifted by 1 position because Dream-family models predict next-token
  (position-i logit predicts position-(i+1) token), not same-position (BERT-style).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)


def shift_logits_right(logits: torch.Tensor) -> torch.Tensor:
    """Shift logits right by 1 along sequence dimension.

    Required for Dream-family models: position-i logit predicts position-(i+1) token.
    Same convention as CDLM's `shift_tensors` (gated on `enable_shift=true` for Dream).
    """
    # logits: [B, L, V]
    # After shift: position 0 stays the same; position i (i>0) gets logits from position i-1.
    return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)


@dataclass
class MaskedDiffusionCollator:
    """Apply biased mask ratio + padding for masked diffusion training.

    For each example, samples mask_ratio = uniform()^bias.
    bias < 1 biases toward HIGH ratios (hard cases, important for low-step decoding).
    """
    tokenizer: Any
    mask_id: int
    pad_id: int
    max_seq_length: int
    mask_ratio_bias: float = 0.3

    def __call__(self, features):
        batch_size = len(features)
        max_len = min(self.max_seq_length, max(len(f["input_ids"]) for f in features))

        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, f in enumerate(features):
            ids = f["input_ids"][:max_len]
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :n] = 1

            # Sample biased mask ratio
            mask_ratio = random.random() ** self.mask_ratio_bias
            mask_ratio = max(0.05, min(0.95, mask_ratio))  # clamp for stability

            # Apply mask only within the answer/response region (last 256 tokens)
            # Diffusion training masks the answer, not the prompt
            ans_start = max(0, n - 256)
            answer_len = n - ans_start
            n_to_mask = max(1, int(answer_len * mask_ratio))

            # Random positions within the answer to mask
            positions = list(range(ans_start, n))
            random.shuffle(positions)
            mask_positions = positions[:n_to_mask]

            for pos in mask_positions:
                labels[i, pos] = input_ids[i, pos].item()  # save target
                input_ids[i, pos] = self.mask_id           # mask in input

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DreamDiffusionTrainer(Trainer):
    """Custom Trainer with Dream-style masked-diffusion loss.

    Key detail: Dream models predict position-(i+1) token from position-i logit,
    so logits must be shifted right by 1 before computing CE against labels.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # kwargs absorbs num_items_in_batch (transformers >= 4.46) and other future args
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, L, V]

        # Dream convention: shift logits right by 1
        # so logits[:, i] now represents prediction for position i (was position i+1)
        shifted_logits = shift_logits_right(logits)

        # CE loss only on masked positions (labels != -100)
        loss = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        return (loss, outputs) if return_outputs else loss


def load_train_data(data_dir: Path):
    """Load JSONL prepared by data_prep.py."""
    return load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    config = OmegaConf.load(args.config)

    base_model_path = config.paths.base_model
    data_dir = Path(config.paths.data_dir)
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Load tokenizer ===
    print(f"Loading tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    mask_id = tokenizer.convert_tokens_to_ids("<|mask|>")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    if mask_id is None or mask_id == tokenizer.unk_token_id:
        raise ValueError(
            "Could not resolve <|mask|> token id from tokenizer. "
            f"Got: {mask_id}. Expected 151666 for Dream-Coder."
        )
    print(f"  mask_id = {mask_id}, pad_id = {pad_id}")

    # === Load model (full FT, no LoRA) ===
    print(f"Loading model from {base_model_path}")
    model = AutoModel.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    # === Load training data ===
    print(f"Loading training data from {data_dir}")
    train_dataset = load_train_data(data_dir)
    print(f"  Loaded {len(train_dataset)} samples")

    # === Data collator ===
    collator = MaskedDiffusionCollator(
        tokenizer=tokenizer,
        mask_id=mask_id,
        pad_id=pad_id,
        max_seq_length=config.training.max_seq_length,
        mask_ratio_bias=config.distillation.mask_ratio_bias,
    )

    # === Training args ===
    # Single A100 80GB: no DeepSpeed needed, full FT fits with 8-bit Adam + grad_ckpt
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.per_device_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.scheduler,
        weight_decay=config.training.weight_decay,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optimizer,
        max_grad_norm=config.training.max_grad_norm,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        logging_steps=config.training.logging_steps,
        report_to=("wandb" if config.logging.wandb_enabled else config.logging.report_to),
        save_strategy="steps",
        save_safetensors=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    # === Train ===
    trainer = DreamDiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        # Use processing_class (new API in transformers >= 4.46), fallback to tokenizer
        **(
            {"processing_class": tokenizer}
            if hasattr(Trainer, "_inner_training_loop")
            else {"tokenizer": tokenizer}
        ),
    )

    print("Starting training ...")
    trainer.train()

    # Save final
    final_path = output_dir / "checkpoints" / "best"
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
