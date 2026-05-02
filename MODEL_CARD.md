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

## Findings — what we learned (factual)

### About the recipe

1. **Bias = 0.3 is too gentle.** It shifted the mask-ratio distribution but ~12% of training samples were still in the low-mask regime, so the model kept reinforcing "commit one token per step" behavior. Recipe moved the model in the right direction by a fraction of what was needed.

2. **`max_grad_norm = 1.0` was a meaningful bottleneck.** Pre-clip gradient norms ran 50–100 throughout training. Clipping at 1.0 means we were scaling gradient magnitudes down by ~50–100×, so the *effective* learning rate was much smaller than the nominal `1.5e-6`.

3. **781 optimizer steps may have been undertrained.** Reported loss never visibly converged — bounced in `[11.2, 13.1]` for the entire run. Whether that's true undertraining or noisy reporting we cannot distinguish from the loss alone, but more training would not have hurt.

4. **The 16-step "sweet spot" was incidental.** We targeted 4 steps and got nothing meaningful there. The largest Δ (+0.04) landed at 16 steps, which we did not aim at and which is only 2× faster than vanilla. The recipe produced a *uniform* small shift, not the *targeted* low-step shift we wanted.

### About measurement

5. **`limit=20` evaluations are misleading.** The scout run showed +0.150 at 8 steps with limit=20 problems. The full eval at limit=50 showed +0.020. The "+0.150" was within sampling noise of a 20-trial measurement. **Minimum trustworthy size for pass@1: limit=50; ideally limit=164 (full HumanEval+).**

6. **HF Trainer reported loss with grad_accum is uninterpretable on its own.** With `model_accepts_loss_kwargs=False` and `compute_loss_func=None`, the reported `loss` is per-microbatch mean CE. With biased mask sampling, that's noise-dominated and bounces by ±1.5 between log lines. **For diffusion-style training, downstream eval is the only signal.**

7. **Per-token CE near `ln(vocab)` does not mean "random model".** At very high mask ratios (90%+ masked), CE ~12 is the natural floor regardless of model quality. Don't read the loss number as a quality measure when training with high mask bias.

### About the infrastructure (gotchas worth documenting)

8. **Several public HF "code" datasets are streaming-incompatible.** `HuggingFaceTB/stack-edu` and `HuggingFaceTB/smollm-corpus` (`python-edu` config) expose only metadata (`blob_id`, `path`, `score`) when streamed — the actual code lives in S3 and must be fetched separately by `blob_id`. A streaming pass over 3M+ rows will yield 0 collected samples.

9. **`Dream-org/Dream-Coder-RL-17k` is prompts-only.** It has no `response`/`solution`/`answer` field; it's RL data where the model generates and a sandbox grades. Cannot be used for SFT-style replay.

10. **`Trainer.save_model()` is not self-contained for `trust_remote_code` models.** It saves weights + `config.json` but not the custom `modeling_*.py` / `configuration_*.py` / `tokenization_*.py` files. `AutoModel.from_pretrained` then fails on load. We patched our `train.py` to copy `.py` files from the base model directory after `save_model`.

11. **HF Trainer's launcher banner doesn't read the YAML.** A hardcoded `Effective batch: 64 (1 × 64)` printed in `02_train.sh` regardless of the actual `per_device_batch_size` / `gradient_accumulation_steps`. We fixed the script to parse and print the real values + warn if the effective batch deviates from the scout-validated 64.

### About the hardware

12. **Initial wall-time projection (12-20 h) was 4× too pessimistic.** Actual training: **3 h 53 min**. The overestimate came from assuming `per_device_batch_size=1` + `gradient_checkpointing=true` throughout. Switching to `batch=8` + `grad_ckpt=false` mid-experiment cut step time from ~20 s to ~17 s.

13. **VRAM was under-utilized at the chosen config.** Final settings used 62 GB / 80 GB. `batch=12` would fit in ~70 GB and give ~17% additional speedup at no quality cost. Not worth restarting this run; useful knob for next iteration.

14. **Cost was 1/3 of the projected $15-25.** Actual end-to-end: **~$5** (setup + scout + full train + validate). The savings came from faster wall time, simpler data pipeline, and avoiding wasted retries.

## What to try in the next iteration

In rough order of expected impact:

1. **`max_grad_norm: 5.0`** (was 1.0) — let real gradient signal through. Single biggest unused lever.
2. **`mask_ratio_bias: 0.15-0.20`** (was 0.3) — push more samples into the high-mask regime.
3. **`learning_rate: 3e-6`** (was 1.5e-6) — combined with looser clipping, should produce visible loss decrease.
4. **More training** — `num_epochs: 2-3` or `total_target_samples: 100000`.
5. **Eval `limit ≥ 50` minimum**, ideally `limit=164` (full HumanEval+) — small evals give misleading scout signals.

These changes are independent. A safer bisection: try (1)+(2) first as the "loosen the recipe" pass and check that 32-step quality doesn't break, then layer on (3)+(4).

## What this checkpoint IS and ISN'T

**Is:**
- A reproducible Dream-Coder fine-tune that's slightly better than vanilla at low step counts (`+0.02` at 4-8 steps, `+0.04` at 16 steps).
- A working data point showing bias=0.3 + LR=1.5e-6 + 1 epoch on 50k samples is undertuned for the 4-step target.
- A correctness-validated training pipeline: matches `Dream-Coder/instruct/src/trainer/fsdp_sft_trainer.py` for loss, attention shape, position_ids, and logit shift.

**Isn't:**
- A 4-step Dream-Coder. The original goal was not achieved.
- A model worth deploying over vanilla in serious applications — gains are ~within evaluation noise for end users.
- An invalidation of the bias-mask method. We tested one configuration; the negative result is on this configuration, not on the approach.

## Reproduce

Training code: https://github.com/modhisathvik7733/dream-coder-fast-pretrain

## License

- Code: MIT.
- Weights: inherited from `Dream-org/Dream-Coder-v0-Instruct-7B` (Apache-2.0).
- Training data: `inclusionAI/Ling-Coder-SFT` (per its license).
