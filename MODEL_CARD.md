---
license: apache-2.0
base_model: Dream-org/Dream-Coder-v0-Instruct-7B
tags:
  - code
  - python
  - diffusion
  - masked-diffusion
  - low-step-decoding
language:
  - en
library_name: transformers
---

# Dream-Coder Fast 7B (Instruct) — experiment checkpoint

Continued pretraining of [Dream-Coder-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-Coder-v0-Instruct-7B) with biased mask-ratio sampling, intended to enable low-step block decoding.

**Status: working but modest gains.** This checkpoint is published as a reproducible data point. The original goal of "4 denoising steps at near-baseline quality" was not achieved at this configuration.

## Results — HumanEval+ pass@1, limit=50 problems

| steps/block | trained | baseline (vanilla) | Δ |
|---:|---:|---:|---:|
| 2 | 0.220 | 0.220 | 0.000 |
| 4 | 0.380 | 0.360 | +0.020 |
| 8 | 0.540 | 0.520 | +0.020 |
| 16 | **0.620** | 0.580 | **+0.040** |
| 32 | 0.740 | 0.760 | −0.020 |

Best operating point: **16 steps/block** (~2× faster than vanilla 32-step with +4 pts quality).

## What changed

- Same architecture as Dream-Coder-Instruct-7B — drop-in replacement.
- Retrained on its own Stage-2 SFT data (`inclusionAI/Ling-Coder-SFT`) for 1 epoch with biased mask-ratio sampling: `t = uniform(0,1)^0.3` instead of `uniform(0,1)`.
- Loss / forward / shift conventions match `Dream-Coder/instruct/src/trainer/fsdp_sft_trainer.py` verbatim.

## Training

| Setting | Value |
|---|---|
| Hardware | 1× A100-SXM4-80GB |
| Optimizer | AdamW 8-bit (bitsandbytes) |
| LR | 1.5e-6 (cosine, 5% warmup) |
| Effective batch | 64 (8 × 8 grad-accum) |
| Sequence length | 768 |
| Mask-ratio bias | 0.3 |
| max_grad_norm | 1.0 |
| Steps | 781 (1 epoch on 50k samples) |
| Wall time | ~3 h 53 min |

## Inference

```python
from transformers import AutoModel, AutoTokenizer
import torch

REPO = "ModhiSathvik/dream-coder-fast-7b-instruct"

model = AutoModel.from_pretrained(
    REPO, torch_dtype=torch.bfloat16, trust_remote_code=True
).to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)

prompt = (
    "<|im_start|>user\n"
    "Write a Python function that returns the n-th Fibonacci number.\n"
    "<|im_end|>\n<|im_start|>assistant\n"
)
inp = tokenizer(prompt, return_tensors="pt").to("cuda")

# Recommended: 16 steps/block — best speed/quality from this checkpoint
out = model.diffusion_generate(
    inp.input_ids,
    attention_mask=inp.attention_mask,
    max_new_tokens=256,
    steps=16 * (256 // 32),
    temperature=0.2,
    top_p=0.95,
    alg="entropy",
)
print(tokenizer.decode(out.sequences[0][inp.input_ids.shape[1]:], skip_special_tokens=True))
```

Step-count guidance:

| `steps` arg | Speed | Quality |
|---|---|---|
| `4 * (max_new_tokens // 32)` | ~8× | ~0.38 (slight win over baseline@4) |
| `8 * (max_new_tokens // 32)` | ~4× | ~0.54 |
| `16 * (max_new_tokens // 32)` (recommended) | ~2× | **~0.62 (best Δ vs baseline)** |
| `32 * (max_new_tokens // 32)` | 1× | ~0.74 (slight regression vs baseline@32) |

## What to try in the next iteration

- bias=0.15-0.20 (more aggressive — push more samples into the high-mask regime)
- max_grad_norm=5.0 (loosen gradient clipping)
- 2-3 epochs or 100k samples (more training signal — current 781 steps may have been undertrained)
- Higher LR (3e-6) combined with looser clipping

## Reproduce

Training code: https://github.com/modhisathvik7733/dream-coder-fast-pretrain

## License

- Code: MIT.
- Weights: inherited from `Dream-org/Dream-Coder-v0-Instruct-7B` (Apache-2.0).
- Training data: `inclusionAI/Ling-Coder-SFT` (per its license).
