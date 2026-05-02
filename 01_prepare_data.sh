#!/usr/bin/env bash
# Prepare training data: KodCode + OpenCodeReasoning (Python replay)
# Plus subset of Stack v2 if available
set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$DATA_DIR"

echo "=== Preparing training data ==="
python "$SCRIPT_DIR/src/data_prep.py" \
    --output_dir "$DATA_DIR" \
    --target_samples 50000 \
    --max_seq_length 768

echo
echo "Data ready at $DATA_DIR"
ls -la "$DATA_DIR"
echo
echo "Next: bash 02_train.sh"
