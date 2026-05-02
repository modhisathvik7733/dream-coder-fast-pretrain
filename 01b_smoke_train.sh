#!/usr/bin/env bash
# Smoke test for the training pipeline before committing to a 12-20 hour run.
#
# What this catches BEFORE you waste GPU hours:
#   - tokenizer / mask_id mismatch
#   - dataset schema mismatch (prompt_length missing, etc.)
#   - model.forward signature mismatch (4D attention mask, position_ids)
#   - logit shift correctness (loss should be in a sane range, not NaN)
#   - bnb 8-bit Adam available (your bitsandbytes install)
#   - gradient checkpointing + bf16 actually fit on this GPU
#   - peak VRAM is below 80 GB so we have margin for the real run
#   - real training step time so you can sanity-check the 12-20h estimate
#
# Runs 5 optimizer steps (5 * grad_accum_steps forward/backward calls).
# Expected duration: ~5-10 minutes on 1× A100 SXM 80GB.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$WORKSPACE/data/train.jsonl" ]; then
    echo "ERROR: training data not found at $WORKSPACE/data/train.jsonl"
    echo "Run bash 01_prepare_data.sh first."
    exit 1
fi

# Verify the new schema (with prompt_length) — old data without it should be regenerated.
python3 - <<'PY'
import json, sys
with open("/workspace/data/train.jsonl") as f:
    first = json.loads(f.readline())
required = {"input_ids", "attention_mask", "prompt_length", "length"}
missing = required - set(first.keys())
if missing:
    print(f"ERROR: train.jsonl missing fields {missing}.")
    print("Old data format detected. Re-run: bash 01_prepare_data.sh")
    sys.exit(1)
print(f"OK: data schema has {sorted(first.keys())}")
print(f"     first sample: prompt_length={first['prompt_length']}, length={first['length']}")
PY

echo
echo "=== Smoke test: 5 training steps (NOT a real run) ==="
echo "  Validates: pipeline correctness, VRAM headroom, step time"
echo "  Real training: bash 02_train.sh (after this passes)"
echo

START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file "$SCRIPT_DIR/configs/acc_config" \
    --num_processes 1 \
    "$SCRIPT_DIR/src/train.py" \
    --config "$SCRIPT_DIR/configs/training.yaml" \
    --max_steps 5 \
    --output_dir_override "$WORKSPACE/smoke_test_output"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo
echo "=== Smoke test passed (${ELAPSED}s) ==="

# Check peak VRAM via nvidia-smi (best-effort).
if command -v nvidia-smi >/dev/null 2>&1; then
    USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
    TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    echo "  GPU memory at end: ${USED_MB} MB / ${TOTAL_MB} MB"
    echo "  (Note: peak is higher mid-step. Watch nvidia-smi during real run.)"
fi

# Clean up smoke output (we don't need it).
rm -rf "$WORKSPACE/smoke_test_output"

echo
echo "If loss looked reasonable (decreasing, no NaN) and VRAM stayed below 75 GB,"
echo "you're cleared for the real run:"
echo
echo "    bash 02_train.sh"
echo
