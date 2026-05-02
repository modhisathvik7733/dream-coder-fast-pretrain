# Fast Dream-Coder — Continued Pretraining for Low-Step Block Decoding

Trains Dream-Coder-v0-Instruct-7B to natively produce good output at **4 denoising steps per 32-token block** instead of the default 32 steps. Result: ~8× faster inference with quality preservation.

**Key property:** continued training (e.g., adding web/multi-language capability) is just standard SFT after this — no adapter management, no consistency-property loss.

## Pipeline overview

```
Phase A (this repo):  Continued pretrain → fast Dream-Coder
                                              │
                                              ▼
Phase B (later):      SFT for new languages/domains (web, Java, etc.)
                                              │
                                              ▼
Phase C (later):      Train small draft + DFlash speculative decoding
```

## Hardware

Designed for: **4× A100 PCIe 40GB** (vast.ai listing types).
- Memory: DeepSpeed ZeRO Stage 3 + 8-bit AdamW + gradient checkpointing
- Per-GPU usage: ~30-35 GB during training
- PCIe 3.0 works (slower); PCIe 4.0 preferred

Single A100 80GB also works (modify configs/acc_config to `num_processes: 1`).

## Quick start (after renting on vast.ai)

```bash
# On the rented instance:
git clone https://github.com/<your_user>/<this_repo>.git
cd <this_repo>/fast-pretrain

bash 00_setup.sh          # ~10 min — install deps + download Dream-Coder-Instruct
bash 01_prepare_data.sh   # ~30 min — pull KodCode + OpenCodeReasoning + format
bash 02_train.sh          # ~50-90 hours — continued pretraining
bash 03_validate.sh       # ~30 min — multi-step-count benchmark
```

Or run training in background via tmux:
```bash
bash monitor.sh
tmux attach -t fastpretrain
```

## Configuration

All hyperparameters in `configs/training.yaml`:

```yaml
training:
  learning_rate: 1.5e-6           # very low for full FT
  num_epochs: 1                   # one pass over data
  per_device_batch_size: 1
  gradient_accumulation_steps: 32 # effective batch 128

distillation:
  teacher_steps_per_block: 32     # vanilla Dream-Coder
  student_steps_per_block: 4      # target speed
  block_size: 32
  mask_ratio_bias: 0.3            # biased toward high-mask training

data:
  python_replay_ratio: 0.4        # anti-forgetting
  reasoning_ratio: 0.3            # preserves R1 distillation
  general_code_ratio: 0.3
  total_target_samples: 50000     # ~200M tokens
```

## Method explained

**Standard masked-diffusion training:** sample mask ratio uniformly in [0, 1]. The model learns to predict masked tokens given visible context, across all difficulty levels.

**This pipeline:** bias mask ratio toward HIGH values (more masking per step). High-mask cases correspond to "early denoising" — many tokens still masked, requiring big confident jumps. Training predominantly on these cases teaches the model to commit many tokens per step, enabling 4-step block decoding.

Loss is standard cross-entropy on masked positions, weighted by mask schedule.

## Expected results

| Config | Pass@1 (HumanEval+) | Wall-clock per problem | Speedup |
|---|---|---|---|
| Vanilla Dream-Coder, 32 steps/block | ~78% | ~7-12s | 1× |
| **After this training, 4 steps/block** | **~75-78%** | **~1-1.5s** | **~6-8×** |
| After this training, 2 steps/block | ~70-74% | ~0.5-0.8s | ~12-16× |

Quality at 4 steps/block should match vanilla within 3 points.

## Continued training after Phase A

Standard SFT on the produced checkpoint. Example (web/multi-lang):

```bash
# Use this checkpoint as base for further SFT
python train_phase_b.py \
    --base_model /workspace/fast_pretrain_output/checkpoints/best \
    --data new_capability_data.jsonl \
    --learning_rate 1.0e-6 \
    --num_epochs 1 \
    --replay_ratio 0.15
```

The "fast" property baked into base weights generally preserves under SFT. Validate periodically with `03_validate.sh` to confirm quality at low step counts holds.

## Cost estimate

Hardware: 4× A100 PCIe 40GB at vast.ai prices.

| Phase | Hours | Cost @ $1.685/hr | Cost @ $3.23/hr |
|---|---|---|---|
| Setup | 0.5 | $1 | $2 |
| Data prep | 0.5 | $1 | $2 |
| Training | 50-90 | $84-152 | $162-291 |
| Validation | 1 | $2 | $3 |
| Buffer | 5 | $9 | $16 |
| **Total** | ~60-100 | **~$97-165** | **~$185-314** |

Recommended: $1.685/hr listing if available (PCIe 3.0, slower training). $3.23/hr for faster.

## Troubleshooting

### OOM during training
```yaml
# In configs/training.yaml, reduce batch and increase accumulation:
per_device_batch_size: 1
gradient_accumulation_steps: 64   # was 32
```

### Loss not decreasing
- Check tokenization: `python src/data_prep.py --target_samples 100` should produce coherent samples
- Verify mask_id = 151666 for Dream-Coder
- Lower LR to 1e-6 or 5e-7

### Quality regression vs baseline
- Increase `python_replay_ratio` from 0.4 to 0.5
- Lower `mask_ratio_bias` from 0.3 to 0.5 (less aggressive high-masking)
- Reduce `num_epochs` to 0.5

### Speed not improved at low step counts
- Increase `mask_ratio_bias` toward 0.2 (more aggressive high-masking)
- Train longer (more samples)
- Verify model has gradient flow at low-step inference (sanity check tokens predicted)

## File structure

```
fast-pretrain/
├── README.md
├── 00_setup.sh             Environment + model download
├── 01_prepare_data.sh      Dataset preparation launcher
├── 02_train.sh             Training launcher
├── 03_validate.sh          Multi-step-count benchmark
├── monitor.sh              tmux session helper
├── configs/
│   ├── training.yaml       All hyperparameters
│   ├── ds_zero3.yaml       DeepSpeed ZeRO Stage 3 config
│   └── acc_config          accelerate config
└── src/
    ├── data_prep.py        Pull KodCode + OpenCodeReasoning + format
    ├── train.py            Custom Trainer with biased-mask diffusion loss
    └── eval_speed_quality.py  Benchmark across step counts
```

## License

Code: MIT.
Model weights: per Dream-org/Dream-Coder-v0-Instruct-7B license.
Datasets: per their respective licenses (KodCode, OpenCodeReasoning, Bespoke-Stratos).
