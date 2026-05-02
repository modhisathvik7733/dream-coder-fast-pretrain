#!/usr/bin/env bash
# One-shot setup for the fast_dllm smoke benchmark.
# Run this on a fresh vast.ai instance (or any GPU box with ≥24 GB VRAM).
#
# What it does:
#   1. Installs required Python packages
#   2. Clones the official Dream-Coder repo (for src.inference.fast_dllm code)
#   3. Downloads Dream-Coder-v0-Instruct-7B (~14 GB)
#   4. Verifies CUDA + tokenizer mask_id sanity
#
# Disk: ~30 GB needed. VRAM: ~16 GB needed. Time: ~5-10 min.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
DREAM_REPO="${DREAM_REPO:-$WORKSPACE/Dream-Coder}"
MODEL_DIR="${MODEL_DIR:-$WORKSPACE/models/dream-coder-7b-instruct}"

mkdir -p "$WORKSPACE" "$WORKSPACE/models"

echo "=== System check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv | head -2
df -h "$WORKSPACE" 2>/dev/null || df -h /
echo

# ---- 1. Python deps ----
echo "=== [1/3] Installing Python packages ==="
pip install --upgrade pip
pip install \
    "transformers==4.48.3" \
    accelerate \
    huggingface_hub \
    hf_transfer

# ---- 2. Clone Dream-Coder for fast_dllm code ----
echo
echo "=== [2/3] Cloning Dream-Coder repo (for fast_dllm code) ==="
if [ -d "$DREAM_REPO/.git" ]; then
    echo "Already cloned at $DREAM_REPO"
    (cd "$DREAM_REPO" && git pull --ff-only) || true
else
    git clone --depth 1 https://github.com/DreamLM/Dream-Coder.git "$DREAM_REPO"
fi

# Sanity-check the fast_dllm path exists where we expect
FAST_DLLM_DIR="$DREAM_REPO/instruct/src/inference/fast_dllm"
if [ ! -f "$FAST_DLLM_DIR/generation_utils_block.py" ]; then
    echo "ERROR: $FAST_DLLM_DIR/generation_utils_block.py not found."
    echo "       The Dream-Coder repo layout may have changed."
    exit 1
fi
echo "  Confirmed: $FAST_DLLM_DIR/generation_utils_block.py"

# ---- 3. Download model ----
echo
echo "=== [3/3] Downloading Dream-Coder-v0-Instruct-7B (~14 GB) ==="
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ -f "$MODEL_DIR/config.json" ]; then
    echo "Model already at $MODEL_DIR — skipping download"
else
    huggingface-cli download Dream-org/Dream-Coder-v0-Instruct-7B \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
fi

# ---- Smoke check ----
echo
echo "=== Smoke check — load tokenizer + verify mask_id ==="
python <<EOF
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$MODEL_DIR", trust_remote_code=True)
mask_id = tok.convert_tokens_to_ids("<|mask|>")
assert mask_id == 151666, f"mask_id mismatch: got {mask_id}"
print(f"  mask_id = {mask_id}  OK")
EOF

echo
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo "Run the benchmark next:"
echo "  python bench/fast_dllm_smoke.py \\"
echo "      --dream_repo $DREAM_REPO \\"
echo "      --model_path $MODEL_DIR"
echo
