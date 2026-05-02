#!/usr/bin/env bash
# Phase A: Continued pretraining for low-step block decoding.
# Trains Dream-Coder to produce good output at 4 steps/block instead of 32.
# ~50-90 hours on 4× A100 PCIe 40GB depending on PCIe version.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sanity check
if [ ! -d "$WORKSPACE/data" ] || [ -z "$(ls -A "$WORKSPACE/data" 2>/dev/null)" ]; then
    echo "ERROR: training data not found at $WORKSPACE/data"
    echo "Run bash 01_prepare_data.sh first."
    exit 1
fi

echo "=== Continued pretraining: Dream-Coder for low-step decoding ==="
echo "  Hardware: 4× A100 PCIe 40GB (DeepSpeed ZeRO Stage 3)"
echo "  Target: 4 denoising steps per 32-token block (vs 32 baseline)"
echo "  Method: standard CE loss with biased mask ratio (bias=0.3, favors high-mask)"
echo "  Includes: Dream-style logit shift (position-i logit predicts position-(i+1) token)"
echo "  Effective batch: 128 (4 GPUs × 1 batch × 32 grad_accum)"
echo "  Expected duration: 50-90 hours"
echo

START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file "$SCRIPT_DIR/configs/acc_config" \
    --num_processes 4 \
    --main_process_port 29520 \
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
