"""Continued pretraining of Dream-Coder for low-step block decoding.

Mechanism (masked diffusion training with biased mask ratio):
  Standard masked-diffusion training samples mask ratios uniformly in [0, 1].
  Lower ratios = late denoising (few tokens still masked, easy task).
  Higher ratios = early denoising (most tokens masked, hard task — needs big jumps).

  To teach the model to make BIGGER jumps per step (so 4 steps suffice instead of 32),
  we bias the mask ratio toward HIGH values (t = uniform()^bias with bias=0.3).
  The model trains predominantly on hard high-mask-ratio cases, learning to commit
  many tokens at once confidently.

This module implements the Dream-Coder SFT training conventions verbatim, with
the SOLE substantive change being the biased `t` distribution. Conventions copied
from Dream-Coder/instruct/src/trainer/{fsdp_sft_trainer.py, sft_dataset.py}:

  1. q_sample-style masking: per-sample t, per-token u; mask where (u < t) AND maskable.
     `maskable_mask` is the loss_mask = 0 on prompt+pad, 1 on response.
  2. attention_mask is all 1s (pad positions also 1) — Dream models attend over
     padding too. Converted 2D -> 4D bidirectional before forward.
  3. position_ids passed explicitly (cumsum of attention_mask - 1).
  4. Logits shifted right by 1: `cat([logits[:,:1], logits[:,:-1]], dim=1)`.
     position-i logit predicts position-(i+1) token (Dream convention).
  5. Loss: CE per-position, masked to t_mask positions only, sum/sum.
     (No time/token reweighting — Dream's are optional optimizations.)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
    Verbatim from Dream-Coder fsdp_sft_trainer.py line 777-779.
    """
    return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)


def biased_q_sample(
    input_ids: torch.LongTensor,
    maskable_mask: torch.BoolTensor,
    mask_token_id: int,
    bias: float,
) -> tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
    """Biased variant of Dream's q_sample (gen_utils.py).

    For each sample, draw t ~ uniform(0, 1)**bias.  Lower bias => more samples
    with high t => more samples in the high-mask regime.  bias=1.0 recovers the
    standard q_sample (uniform t).

    Then per-token: u ~ uniform(0, 1); mask where (u < t) AND maskable.
    Returns (masked_input_ids, t, t_mask).
    """
    B = input_ids.shape[0]
    device = input_ids.device

    t = torch.rand((B,), dtype=torch.float, device=device).pow(bias)
    # Clamp for numerical stability — never fully zero or fully masked.
    t = t.clamp(min=0.05, max=0.95)

    u = torch.rand_like(input_ids, dtype=torch.float)
    t_mask = (u < t[:, None]) & maskable_mask

    masked_input_ids = input_ids.masked_fill(t_mask, mask_token_id)
    return masked_input_ids, t, t_mask


@dataclass
class DreamSFTCollator:
    """Pad + build (input_ids, attention_mask, position_ids, loss_mask, labels).

    Mirrors Dream-Coder's sft_dataset.py conventions:
      - attention_mask is all 1s, INCLUDING on pad positions (so model attends
        across padding — Dream's published training convention).
      - loss_mask = 0 on prompt and pad; 1 on response. This is the maskable
        region for the diffusion objective.
      - position_ids = cumsum(attention_mask) - 1, which is just arange(L)
        because attention_mask is all 1s.

    Masking itself happens inside compute_loss, not here, because q_sample
    needs the model device + per-batch RNG.
    """
    pad_id: int
    max_seq_length: int

    def __call__(self, features):
        B = len(features)
        max_len = min(
            self.max_seq_length,
            max(len(f["input_ids"]) for f in features),
        )

        input_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        # NOTE: attention_mask is initialized to 1 — Dream convention, pad attended too.
        attention_mask = torch.ones((B, max_len), dtype=torch.long)
        loss_mask = torch.zeros((B, max_len), dtype=torch.long)

        for i, f in enumerate(features):
            ids = f["input_ids"][:max_len]
            n = len(ids)
            prompt_len = min(int(f.get("prompt_length", 0)), n)

            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            # loss_mask: 1 on response tokens [prompt_len, n); 0 on prompt + pad
            if n > prompt_len:
                loss_mask[i, prompt_len:n] = 1

        position_ids = torch.cumsum(attention_mask, dim=-1) - 1
        position_ids = position_ids.clamp(min=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


class DreamDiffusionTrainer(Trainer):
    """Custom Trainer with Dream-style masked-diffusion loss + biased mask ratio.

    Forward call mirrors fsdp_sft_trainer.py::_compute_loss_and_backward:
      - 2D attention_mask -> 4D bidirectional via outer logical_and
      - position_ids passed explicitly
      - logits shifted right by 1
      - CE per-position, masked to actual masked positions, sum/sum
    """
    def __init__(self, *args, mask_token_id: int, mask_ratio_bias: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_token_id = int(mask_token_id)
        self.mask_ratio_bias = float(mask_ratio_bias)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # kwargs absorbs num_items_in_batch (transformers >= 4.46) and other future args.
        input_ids = inputs["input_ids"]
        attention_mask_2d = inputs["attention_mask"]  # [B, L], all 1s
        position_ids = inputs["position_ids"]
        loss_mask = inputs["loss_mask"].bool()        # [B, L], maskable region

        # Apply q_sample-style biased masking. labels = original input_ids.
        labels = input_ids.clone()
        masked_input_ids, _t, t_mask = biased_q_sample(
            input_ids,
            maskable_mask=loss_mask,
            mask_token_id=self.mask_token_id,
            bias=self.mask_ratio_bias,
        )

        # 2D -> 4D bidirectional attention mask (Dream convention,
        # fsdp_sft_trainer.py line 764-767).
        am = attention_mask_2d.bool()
        attention_mask_4d = torch.logical_and(
            am.unsqueeze(1).unsqueeze(-2),
            am.unsqueeze(1).unsqueeze(-1),
        )  # [B, 1, L, L]

        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = outputs.logits  # [B, L, V]

        # Dream convention: shift logits right by 1
        shifted_logits = shift_logits_right(logits).contiguous()

        V = shifted_logits.size(-1)
        loss_per = F.cross_entropy(
            shifted_logits.view(-1, V),
            labels.view(-1),
            reduction="none",
        )
        # Loss only on actually-masked positions (Dream zeroes elsewhere).
        loss_per = loss_per.masked_fill(~t_mask.view(-1), 0.0)

        n_masked = t_mask.sum().clamp(min=1)
        loss = loss_per.sum() / n_masked

        return (loss, outputs) if return_outputs else loss


def load_train_data(data_dir: Path):
    """Load JSONL prepared by data_prep.py."""
    return load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="If >0, override num_epochs and stop after this many steps. Used by smoke test.")
    ap.add_argument("--output_dir_override", type=str, default=None,
                    help="If set, write checkpoints here instead of config.paths.output_dir.")
    args = ap.parse_args()

    config = OmegaConf.load(args.config)

    base_model_path = config.paths.base_model
    data_dir = Path(config.paths.data_dir)
    output_dir = Path(args.output_dir_override or config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Load tokenizer ===
    print(f"Loading tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    mask_id = tokenizer.convert_tokens_to_ids("<|mask|>")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

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
    collator = DreamSFTCollator(
        pad_id=pad_id,
        max_seq_length=config.training.max_seq_length,
    )

    # === Training args ===
    # Single A100 80GB: no DeepSpeed needed, full FT fits with 8-bit Adam + grad_ckpt.
    ta_kwargs = dict(
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
    if args.max_steps > 0:
        ta_kwargs["max_steps"] = args.max_steps
        ta_kwargs["save_strategy"] = "no"   # smoke test: don't bother saving
    training_args = TrainingArguments(**ta_kwargs)

    # === Train ===
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        mask_token_id=mask_id,
        mask_ratio_bias=config.distillation.mask_ratio_bias,
    )
    # Use processing_class (transformers >= 4.46), else tokenizer kwarg.
    if hasattr(Trainer, "_inner_training_loop"):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DreamDiffusionTrainer(**trainer_kwargs)

    print("Starting training ...")
    trainer.train()

    # Save final (skip if smoke test)
    if args.max_steps <= 0:
        final_path = output_dir / "checkpoints" / "best"
        final_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        print(f"Saved final checkpoint to {final_path}")
    else:
        print(f"Smoke test complete ({args.max_steps} steps). No checkpoint saved.")


if __name__ == "__main__":
    main()
