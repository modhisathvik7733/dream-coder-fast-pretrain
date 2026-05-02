#!/usr/bin/env bash
# Phase A: Continued pretraining for low-step block decoding.
# Trains Dream-Coder to produce good output at 4 steps/block instead of 32.
#
# Hardware target: 1× A100 SXM 80GB (single GPU, no DeepSpeed).
# Expected duration: ~10-20 hours for 30k samples.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sanity check
if [ ! -f "$WORKSPACE/data/train.jsonl" ]; then
    echo "ERROR: training data not found at $WORKSPACE/data/train.jsonl"
    echo "Run bash 01_prepare_data.sh first."
    exit 1
fi

# Schema check — old data without prompt_length will fail silently otherwise.
python3 - <<'PY' || { echo "Re-run: bash 01_prepare_data.sh"; exit 1; }
import json, sys
with open("/workspace/data/train.jsonl") as f:
    sample = json.loads(f.readline())
required = {"input_ids", "attention_mask", "prompt_length", "length"}
missing = required - set(sample.keys())
if missing:
    print(f"ERROR: train.jsonl missing fields {missing}.")
    sys.exit(1)
PY

# Strongly suggest the smoke test if it hasn't been run.
echo "REMINDER: have you run 'bash 01b_smoke_train.sh' yet?"
echo "  It catches pipeline bugs in 5 minutes instead of 12 hours."
echo "  Press Ctrl-C now to bail out, or wait 10s to continue."
sleep 10

# Disk space check (training will save ~14GB checkpoints)
DISK_FREE_GB=$(df -BG /workspace 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G' || echo "0")
echo "Disk free: ${DISK_FREE_GB}GB"
if [ "${DISK_FREE_GB:-0}" -lt 50 ]; then
    echo "WARNING: less than 50GB free disk. Training may fail at checkpoint save."
    echo "  With save_total_limit=2 and 14GB/checkpoint, you need 28GB+ for checkpoints alone."
    echo "  Recommend: clear /workspace/data/cache or other temp files before continuing."
fi

# Read the ACTUAL batch / accum values from training.yaml so the banner doesn't lie.
PER_DEV_BATCH=$(grep -E '^[[:space:]]*per_device_batch_size:' "$SCRIPT_DIR/configs/training.yaml" | awk '{print $2}')
GRAD_ACCUM=$(grep -E '^[[:space:]]*gradient_accumulation_steps:' "$SCRIPT_DIR/configs/training.yaml" | awk '{print $2}')
GRAD_CKPT=$(grep -E '^[[:space:]]*gradient_checkpointing:' "$SCRIPT_DIR/configs/training.yaml" | awk '{print $2}')
EFF_BATCH=$(( PER_DEV_BATCH * GRAD_ACCUM ))

echo
echo "=== Continued pretraining: Dream-Coder for low-step decoding ==="
echo "  Hardware: 1× A100 SXM 80GB (single GPU, no DeepSpeed)"
echo "  Target: 4 denoising steps per 32-token block (vs 32 baseline)"
echo "  Method: q_sample-style biased masking (bias=0.3, favors high-mask)"
echo "  Conventions: Dream-Coder SFT trainer verbatim — 4D attention mask,"
echo "               position_ids, loss_mask=response-only, shifted logits."
echo "  Effective batch: ${EFF_BATCH} (1 GPU × ${PER_DEV_BATCH} batch × ${GRAD_ACCUM} grad_accum)"
echo "  gradient_checkpointing: ${GRAD_CKPT}"
echo "  Samples: 50k × ~700 avg tokens = ~35M tokens"
echo
if [ "${EFF_BATCH}" -ne 64 ]; then
    echo "  WARNING: effective batch is ${EFF_BATCH}, not 64."
    echo "  Scout was validated at effective batch 64. Quality may differ."
    echo "  Press Ctrl-C now to bail and adjust, or wait 10s to continue."
    sleep 10
fi
echo

START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file "$SCRIPT_DIR/configs/acc_config" \
    --num_processes 1 \
    "$SCRIPT_DIR/src/train.py" \
    --config "$SCRIPT_DIR/configs/training.yaml"

END_TIME=$(date +%s)
HOURS=$(( (END_TIME - START_TIME) / 3600 ))
MINUTES=$(( ((END_TIME - START_TIME) % 3600) / 60 ))

echo
echo "=== Training complete ==="
echo "Duration: ${HOURS}h ${MINUTES}m"
echo "Checkpoint: $WORKSPACE/fast_pretrain_output/checkpoints/best"
echo
echo "Next: bash 03_validate.sh"
