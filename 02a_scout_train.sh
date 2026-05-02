#!/usr/bin/env bash
# Scout run: 5k samples, full pipeline end-to-end, then evaluate.
#
# Purpose: catch "is this approach even directionally working" in ~2 hours
# instead of finding out 16 hours into the real run that it's not.
#
# What it does:
#   1. Prepares 5,000 training samples to /workspace/data_scout/
#   2. Trains for 1 epoch (~80 optimizer steps at grad_accum=64)
#   3. Evaluates on 20 HumanEval+ problems at steps={4, 8, 16, 32}
#      against the vanilla baseline at the same step counts
#   4. Prints a go/no-go decision
#
# Wall time:  ~2-3 hours total
# Cost:       ~$2-3 at $0.89/hr
#
# Decision rule:
#   GO  if trained_pass@4 / baseline_pass@32 >= 0.5  AND  trained_pass@32 >= baseline_pass@32 - 0.05
#   NO-GO otherwise — debug bias / data / training schedule before scaling up.
#
# Outputs:
#   /workspace/data_scout/         — 5k samples (kept; full run uses /workspace/data/)
#   /workspace/scout_output/       — checkpoints + validation_results.json
#
# After scout passes, run the full pipeline normally:
#   bash 01_prepare_data.sh    # writes 50k to /workspace/data/
#   bash 02_train.sh           # trains on /workspace/data/
#   bash 03_validate.sh        # validates against baseline

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCOUT_DATA_DIR="$WORKSPACE/data_scout"
SCOUT_OUT_DIR="$WORKSPACE/scout_output"

echo "=== SCOUT RUN — 5k samples, ~2-3h ==="
echo "  Purpose: directional check before the 12-20h real run."
echo "  Data:   $SCOUT_DATA_DIR"
echo "  Output: $SCOUT_OUT_DIR"
echo

# ---- Step 1: prepare 5k samples ----
mkdir -p "$SCOUT_DATA_DIR"
if [ -f "$SCOUT_DATA_DIR/train.jsonl" ]; then
    echo "Reusing existing scout data at $SCOUT_DATA_DIR/train.jsonl"
    echo "  (delete it manually if you want fresh data)"
else
    echo "--- Preparing 5,000 scout samples ---"
    python "$SCRIPT_DIR/src/data_prep.py" \
        --output_dir "$SCOUT_DATA_DIR" \
        --target_samples 5000 \
        --max_seq_length 768
fi
echo

# ---- Step 2: train on the scout data ----
echo "--- Training scout (1 epoch over 5k samples) ---"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file "$SCRIPT_DIR/configs/acc_config" \
    --num_processes 1 \
    "$SCRIPT_DIR/src/train.py" \
    --config "$SCRIPT_DIR/configs/training.yaml" \
    --data_dir_override "$SCOUT_DATA_DIR" \
    --output_dir_override "$SCOUT_OUT_DIR"

END_TIME=$(date +%s)
HOURS=$(( (END_TIME - START_TIME) / 3600 ))
MINUTES=$(( ((END_TIME - START_TIME) % 3600) / 60 ))
echo "Scout training done in ${HOURS}h ${MINUTES}m"
echo

# ---- Step 3: validate against baseline at all step counts ----
SCOUT_CKPT="$SCOUT_OUT_DIR/checkpoints/best"
if [ ! -d "$SCOUT_CKPT" ]; then
    echo "ERROR: scout checkpoint not found at $SCOUT_CKPT"
    exit 1
fi

echo "--- Validating scout (limit=20 problems, steps=4 8 16 32) ---"
python "$SCRIPT_DIR/src/eval_speed_quality.py" \
    --model_path "$SCOUT_CKPT" \
    --baseline_path "$WORKSPACE/models/dream-coder-7b-instruct" \
    --steps_per_block 4 8 16 32 \
    --benchmark humaneval_plus \
    --limit 20 \
    --output "$SCOUT_OUT_DIR/scout_validation.json"

echo
echo "=== SCOUT COMPLETE ==="
echo "Read the SUMMARY table above. Decision rule:"
echo "  GO    — Δ pass@1 at steps=4 is positive (or close to zero)"
echo "          AND Δ pass@1 at steps=32 >= -0.05"
echo "  NO-GO — trained@4 is below baseline@4 by a wide margin,"
echo "          OR trained@32 regressed by more than 5 points."
echo
echo "If GO:    rm -rf $SCOUT_OUT_DIR  &&  bash 01_prepare_data.sh  &&  bash 02_train.sh"
echo "If NO-GO: stop here, share the SUMMARY, debug before spending the full \$15-20."
echo
