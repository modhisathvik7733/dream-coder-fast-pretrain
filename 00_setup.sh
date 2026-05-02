#!/usr/bin/env bash
# Setup for Dream-Coder continued pretraining (low-step block decoding).
# Designed for vast.ai 4× A100 PCIe 40GB instance.
set -euo pipefail

# === Sanity checks ===
echo "=== System check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv | head -8
echo "CPUs: $(nproc)"
df -h /workspace 2>/dev/null || df -h /

WORKSPACE="${WORKSPACE:-/workspace}"
MODEL_DIR="$WORKSPACE/models/dream-coder-7b-instruct"
DATA_DIR="$WORKSPACE/data"
CHECKPOINT_DIR="$WORKSPACE/fast_pretrain_output"

mkdir -p "$WORKSPACE" "$MODEL_DIR" "$DATA_DIR" "$CHECKPOINT_DIR"

# === Install dependencies ===
echo
echo "=== [1/3] Installing Python dependencies ==="
pip install --upgrade pip
pip install \
    "transformers==4.48.3" \
    accelerate \
    peft \
    bitsandbytes \
    deepspeed \
    datasets \
    omegaconf \
    wandb \
    hf_transfer \
    evalplus

# === Download Dream-Coder-Instruct ===
echo
echo "=== [2/3] Downloading Dream-Coder-v0-Instruct-7B (~15 GB) ==="
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ ! -f "$MODEL_DIR/config.json" ]; then
    huggingface-cli download Dream-org/Dream-Coder-v0-Instruct-7B \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
else
    echo "Model already downloaded."
fi

# === Smoke test ===
echo
echo "=== [3/3] Smoke test ==="
python <<EOF
import torch
from transformers import AutoModel, AutoTokenizer
print("Loading tokenizer ...")
tok = AutoTokenizer.from_pretrained("$MODEL_DIR", trust_remote_code=True)
mask_id = tok.convert_tokens_to_ids("<|mask|>")
print(f"  mask_id = {mask_id}")
assert mask_id == 151666, f"mask_id mismatch: got {mask_id}"

print("Loading model (BF16) ...")
m = AutoModel.from_pretrained(
    "$MODEL_DIR", torch_dtype=torch.bfloat16, trust_remote_code=True
).to("cuda:0").eval()
print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

del m
torch.cuda.empty_cache()
print("Smoke test PASSED")
EOF

echo
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo "Next: bash 01_prepare_data.sh"
