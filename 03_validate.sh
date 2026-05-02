#!/usr/bin/env bash
# Validate fast Dream-Coder: HumanEval+ at multiple step counts.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${1:-$WORKSPACE/fast_pretrain_output/checkpoints/best}"

if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo "=== Validating fast Dream-Coder ==="
echo "  Checkpoint: $CHECKPOINT"
echo

# Run benchmark at multiple step counts
python "$SCRIPT_DIR/src/eval_speed_quality.py" \
    --model_path "$CHECKPOINT" \
    --baseline_path "$WORKSPACE/models/dream-coder-7b-instruct" \
    --steps_per_block 2 4 8 16 32 \
    --benchmark humaneval_plus \
    --limit 50 \
    --output "$WORKSPACE/fast_pretrain_output/validation_results.json"

echo
echo "=== Done ==="
echo "Results: $WORKSPACE/fast_pretrain_output/validation_results.json"
echo
echo "Compare baseline (32 steps) vs your distilled model (4-8 steps)."
echo "Quality at 4 steps should be within 3-5 pts of baseline."
