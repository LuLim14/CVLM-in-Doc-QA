# Usage

## Skip training and run evaluation only

`scripts/run_full_pipeline.sh` now honours a `SKIP_TRAIN` env var. Set it to `1`
to skip Step 1/2 and jump straight to checkpoint discovery + evaluation. The
script still needs `OUTPUT_DIR` to point at a run directory that already
contains `model_step_*.safetensors`.

```bash
SKIP_TRAIN=1 \
OUTPUT_DIR=/path/to/existing/run \
bash CVLM/scripts/run_full_pipeline.sh
```

Other eval knobs (`EVAL_MODES`, `EVAL_SPLIT`, `EVAL_MAX_SAMPLES`,
`EVAL_BATCH_SIZE`, `EVAL_COMPUTE_GEN`) behave as before.

## Live PNG training plots (no TensorBoard required)

Both `train_cvlm.py` and `train_sft.py` now write a CSV of training scalars and
render PNG dashboards to disk while training is running. View them with `scp`,
a file browser, or `feh` — no TB process needed.

Outputs in `<output_dir>/`:

| file              | content                                        |
|-------------------|------------------------------------------------|
| `metrics.csv`     | append-only: `step,loss,loss_avg,lr,grad_norm,batch_time` |
| `loss.png`        | loss + running-average loss vs step            |
| `lr.png`          | learning rate vs step                          |
| `grad_norm.png`   | gradient norm vs step (sync-step rows only)    |
| `batch_time.png`  | per-step wall time vs step                     |
| `dashboard.png`   | 2x2 grid of all four plots                     |

New flags (both trainers):

```
--plot_interval N    refresh PNGs every N optimizer steps (default 100; 0 = disable PNGs)
--csv_path PATH      override CSV location (default <output_dir>/metrics.csv)
```

The CSV is always written. Only PNG generation is gated by `--plot_interval`.

## SFT baseline pipeline

Train the same decoder used by CVLM with standard supervised fine-tuning on
`document + "\n\n" + question -> answer`, then evaluate with the same metric
suite as CVLM for direct comparison.

End-to-end:

```bash
OUTPUT_DIR=/path/to/sft_run \
  bash CVLM/scripts/run_sft_pipeline.sh
```

The script: trains `train_sft.py`, saves an HF-format checkpoint to
`OUTPUT_DIR`, then runs `eval_cvlm.py --mode sft --sft_model_path OUTPUT_DIR`
producing `OUTPUT_DIR/eval_sft.json`. JSON keys match a CVLM
`eval_baseline_llm_full.json` so the two are diffable.

Manual training:

```bash
python CVLM/src/train_sft.py \
  --output_dir /path/to/sft_run \
  --dataset_name sggetao/PwC \
  --model_name_or_path HuggingFaceTB/SmolLM-135M-Instruct \
  --epochs 2 --batch_size 2 --lr 1e-5 \
  --max_prompt_len 512 --max_answer_len 1024 \
  --plot_interval 100
```

Manual eval:

```bash
python CVLM/src/eval_cvlm.py \
  --mode sft \
  --sft_model_path /path/to/sft_run \
  --dataset_name sggetao/PwC --dataset_split test \
  --text_encoder_name answerdotai/ModernBERT-base \
  --max_prompt_len 512 --max_answer_len 1024 \
  --max_vision_len 256 --max_source_len 0 --compression_rate 4 \
  --output_json /path/to/sft_run/eval_sft.json \
  --compute_generation_metrics
```

The `--text_encoder_name` and vision-related flags are needed only because the
eval reuses `CvlmTrainDataset` for sample filtering — the SFT decoder itself
ignores the encoder/vision side.

`run_sft_pipeline.sh` env vars:

| var | default | meaning |
|---|---|---|
| `OUTPUT_DIR` | timestamped path | where the HF model + eval JSON go |
| `MODEL_NAME` | `HuggingFaceTB/SmolLM-135M-Instruct` | base decoder to fine-tune |
| `EPOCHS` | 2 | train epochs |
| `BATCH_SIZE` | 2 | per-rank batch |
| `LR` | 1e-5 | AdamW learning rate |
| `MAX_PROMPT_LEN` / `MAX_ANSWER_LEN` | 512 / 1024 | dataset-filter caps |
| `MAX_SAMPLES` | 0 | cap rows; 0 = all |
| `GRAD_ACCUM` | 1 | gradient accumulation |
| `LOG_INTERVAL` | 10 | optimizer steps per CSV/TB log |
| `SAVE_INTERVAL_STEPS` | 500 | HF Trainer save_steps; 0 = end only |
| `PLOT_INTERVAL` | 100 | optimizer steps between PNG refreshes |
| `NPROC` | 1 | torchrun GPUs (≥2 enables DDP) |
| `ENABLE_WARMUP` / `WARMUP_STEPS` | 1 / 100 | linear LR warmup |
| `EVAL_SPLIT` | `test` | split to evaluate on |
| `SKIP_TRAIN` | 0 | set to 1 to skip training and re-eval an existing dir |

## Trackio logging

`SummaryWriter` has been replaced by [trackio](https://github.com/gradio-app/trackio)
for the live dashboard. CSV (`metrics.csv`) and PNG (`dashboard.png`) outputs are
unchanged and continue to work even when trackio is disabled.

### Install

```bash
pip install trackio
```

### Run a training with trackio

```bash
TRACKIO_PROJECT=cvlm \
TRACKIO_RUN_NAME=my-run \
OUTPUT_DIR=/path/to/run \
bash CVLM/scripts/run_full_pipeline.sh
```

Defaults: `TRACKIO_PROJECT=cvlm`, `TRACKIO_RUN_NAME=$(basename "$OUTPUT_DIR")`.
The same env vars are honoured by `scripts/run_sft_pipeline.sh`.

Per-binary CLI flags (equivalent to env vars) are available on
`train_cvlm.py`, `train_sft.py`, and `eval_cvlm.py`:

```
--trackio_project NAME      # default: cvlm
--trackio_run_name NAME     # default: output dir basename
--trackio_space_id user/repo  # optional HF Space to host the dashboard
--trackio_disable           # turn trackio off entirely
```

### View the dashboard

Local UI:

```bash
trackio show --project cvlm
```

Host on HF Spaces (auto-creates the Space the first time):

```bash
TRACKIO_SPACE_ID=username/cvlm-runs bash CVLM/scripts/run_full_pipeline.sh
```

### Disable trackio

```bash
TRACKIO_DISABLE=1 bash CVLM/scripts/run_full_pipeline.sh
# or per-binary:
python CVLM/src/train_cvlm.py --trackio_disable ...
```

CSV + PNG outputs remain available when trackio is disabled.

### Metrics keys

| Phase | Keys |
| --- | --- |
| Train (CVLM) | `train/loss`, `train/loss_avg`, `train/lr`, `train/grad_norm`, `train/batch_time` |
| Train (SFT) | `train/loss`, `train/lr`, `train/grad_norm`, `train/epoch` |
| Eval | `eval/<metric>` (loss, ppl, ROUGE, BLEU-4, EM, compression stats) + `eval/compression_ratio_dist` (histogram if supported, else `*_mean/std/min/max`) |

## Memory optimisations (Liger-Kernel + gradient checkpointing)

### Install Liger-Kernel

CVLM training requires `liger-kernel` (fused Triton kernels for the SmolLM
decoder). Install once into the conda env:

```bash
PYTHONNOUSERSITE=1 /home/jovyan/.mlspace/envs/cvlm/bin/pip install "liger-kernel>=0.8.0,<0.9"
```

Verify:

```bash
PYTHONNOUSERSITE=1 /home/jovyan/.mlspace/envs/cvlm/bin/python -c \
  "from liger_kernel.transformers import apply_liger_kernel_to_llama; \
   import importlib.metadata; \
   print('liger', importlib.metadata.version('liger-kernel'))"
```

Expected: `liger 0.8.x`.

`liger-kernel 0.8.x` requires `torch >= 2.5` (uses the public
`torch.distributed.tensor.DTensor` path). The cvlm env was upgraded to
`torch 2.5.1+cu124` / `triton 3.1.0` for compatibility.

### Gradient checkpointing flag

`scripts/run_full_pipeline.sh` exposes a `GRADIENT_CHECKPOINTING` env var
(default `1`, on). Disable with `GRADIENT_CHECKPOINTING=0` for ~25–35%
faster steps at the cost of ~50% more activation memory:

```bash
GRADIENT_CHECKPOINTING=0 OUTPUT_DIR=/tmp/baseline   bash scripts/run_full_pipeline.sh
GRADIENT_CHECKPOINTING=1 OUTPUT_DIR=/tmp/optimised bash scripts/run_full_pipeline.sh
```

Each run prints `[train] peak GPU memory: X.XX GB` at the end and logs
`system/peak_gpu_gb` to trackio for A/B comparison across runs.

### Direct CLI override

When invoking `src/train_cvlm.py` directly (e.g. via `torchrun`), use
`--gradient_checkpointing true` / `--gradient_checkpointing false`.

## Compression-rate curriculum (in-training)

When CVLM training plateaus at a high static compression rate, train under
an in-training curriculum that starts at a low cr and steps up at fixed
milestones inside one run. Optimizer state is preserved across stages.

### Schedule format

`CR_SCHEDULE` env var (or `--cr_schedule` CLI flag) is a comma-separated
list of `cr:end_step` pairs. Each entry says "use this cr until global_step
reaches end_step". The **last entry's `end_step` MUST be 0** (sentinel for
forever).

Example: cr=1 for steps 0-5999, cr=2 for steps 6000-11999, cr=4 for steps
12000-17999, cr=8 from step 18000 onward:

```bash
CR_SCHEDULE="1:6000,2:12000,4:18000,8:0" \
  OUTPUT_DIR=/path/to/run \
  bash scripts/run_full_pipeline.sh
```

Empty (default) = static cr from `COMPRESSION_RATE`.

### What to watch in trackio

`train/compression_rate` is logged every step; overlay it on `train/loss`
to see exactly when each stage transition happens. Each transition triggers
a forced checkpoint save (`model_step_<N>.safetensors`).

### Eval per-stage

Eval each stage's checkpoint with the cr that was active at its end:

```bash
python src/eval_cvlm.py \
  --checkpoint_path "${OUTPUT_DIR}/model_step_6000.safetensors" \
  --compression_rate 1 \
  --mode cvlm \
  --output_json "${OUTPUT_DIR}/eval_stage1_cr1.json"
```

### Memory note

The cr=1 stage is the heaviest (V == max_source_len, longest decoder
sequence in the curriculum). Set `BATCH_SIZE` to fit the cr=1 stage —
later stages will run with the same batch size and lower memory usage.

## Disable LR schedule (constant LR for curriculum runs)

Cosine annealing over the full step budget kills late curriculum stages —
by the time cr=8 starts, the LR is too low to relearn the bottleneck.
Disable cosine decay so the LR stays constant at `--lr` after warmup:

```bash
DISABLE_LR_SCHEDULE=1 \
  CR_SCHEDULE="1:6000,2:12000,4:18000,8:0" \
  OUTPUT_DIR=/path/to/run \
  bash scripts/run_full_pipeline.sh
```

Equivalent direct CLI: `--disable_lr_schedule true`. Warmup (if enabled)
still ramps linearly; only the post-warmup cosine decay is skipped.

## Eval all checkpoints (per-step trackio scatter)

Sweep every `model_step_*.safetensors` in `OUTPUT_DIR` and log each
checkpoint's metrics into a single per-mode trackio run, indexed by
checkpoint step. The dashboard renders the metric chart as a scatter/line
across checkpoints automatically.

```bash
SKIP_TRAIN=1 \
  EVAL_ALL_CHECKPOINTS=1 \
  OUTPUT_DIR=/path/to/existing/run \
  bash scripts/run_full_pipeline.sh
```

Per-checkpoint JSONs are written as `eval_${MODE}_step${STEP}.json`. The
trackio run name is `${TRACKIO_RUN_NAME}_${MODE}` (one run per mode, N
points per mode at the trained checkpoint steps).

For curriculum runs, evaluate each checkpoint at the cr that was active
when it was saved by also setting `COMPRESSION_RATE` per stage — or run
the sweep multiple times with different `COMPRESSION_RATE` values to
inspect cr-vs-checkpoint generalisation.

## Eval each checkpoint at its native (training-time) cr

For curriculum runs, evaluating every checkpoint at the same fixed cr
mismatches what the model actually learned at each stage. Pass
`EVAL_CR_SCHEDULE` (same format as training `CR_SCHEDULE`) and the eval
sweep will set the model's compression rate per checkpoint based on the
schedule:

```bash
SKIP_TRAIN=1 \
  EVAL_ALL_CHECKPOINTS=1 \
  EVAL_CR_SCHEDULE="1:6000,2:12000,4:18000,8:0" \
  OUTPUT_DIR=/path/to/existing/run \
  bash scripts/run_full_pipeline.sh
```

Each checkpoint at step `S` is evaluated at the cr of the schedule entry
whose `end_step >= S` (i.e. the cr that produced its weights):

| ckpt step | trained at | evaluated at |
|---|---|---|
| ≤ 6000 | cr=1 | cr=1 |
| 6001–12000 | cr=2 | cr=2 |
| 12001–18000 | cr=4 | cr=4 |
| > 18000 | cr=8 | cr=8 |

Per-checkpoint JSONs are written as `eval_${MODE}_native_step${STEP}.json`
(the `_native` suffix avoids overwriting prior fixed-cr eval JSONs in the
same directory). A new scalar `compression_rate` is added to each JSON
and logged to trackio as `eval/compression_rate`, so you can overlay it
on metric scatters to see exactly where each cr stage sits.

`EVAL_CR_SCHEDULE` is only forwarded to cr-dependent modes (`cvlm`,
`cvlm_shuffle`, `baseline_proj`); `baseline_llm` and `baseline_llm_full`
ignore the flag because they don't use compression.

Direct CLI:

```bash
python src/eval_cvlm.py \
  --all_checkpoints \
  --cr_schedule "1:6000,2:12000,4:18000,8:0" \
  --checkpoint_path "${OUTPUT_DIR}/model_step_30201.safetensors" \
  --mode cvlm --compute_generation_metrics \
  --output_json "${OUTPUT_DIR}/eval_cvlm.json"
```

Constraints: `--cr_schedule` requires `--all_checkpoints` and one of the
cr-dependent modes; otherwise eval exits with `ValueError`.

## Unfreeze top-K text-encoder layers

By default the text encoder (ModernBERT-base, 22 layers) is fully frozen.
To unfreeze the top-K transformer layers (plus `final_norm`) and let
gradients flow back into them during training, set `UNFREEZE_ENCODER_TOP_K`
or pass `--unfreeze_encoder_top_k`.

```bash
UNFREEZE_ENCODER_TOP_K=4 \
bash CVLM/scripts/run_full_pipeline.sh
```

Direct CLI:

```bash
python src/train_cvlm.py \
  --output_dir "${OUTPUT_DIR}" \
  --unfreeze_encoder_top_k 4 \
  ...other args...
```

Notes:
- `0` (default) = fully frozen, identical to the legacy code path.
- The top-K layers + `final_norm` are saved in each checkpoint
  (`requires_grad`-filtered safetensors export already handles this — no
  separate flag needed at eval time).
- Old fully-frozen checkpoints load via `strict=False` against a
  `unfreeze_encoder_top_k>0` model: missing top-layer weights will be the
  pretrained ModernBERT init (since we always `from_pretrained` the
  encoder before partial-loading the trainable state dict).
- Gradient checkpointing is enabled on the text encoder automatically
  when `unfreeze_encoder_top_k>0` and `--gradient_checkpointing=1`, to
  bound activation memory through the full encoder stack.
- The encoder is switched to `train()` mode while top-K is unfrozen so
  dropout in the unfrozen blocks is active. Frozen lower layers receive
  no parameter gradients regardless of train/eval state.

## fp32 master weights (numerical-precision control)

Training runs entirely in bf16 and, by default, AdamW updates those bf16
parameters **in place** with bf16 moments (no `torch.autocast`, no master
weights). bf16 keeps 8 mantissa bits, so its **relative granularity is
0.39%-0.78%** (`2^-8`..`2^-7`) of the weight's own magnitude: `ULP = 7.8e-3` at
magnitude ~1.0 and `ULP ~ 1.2e-4` at magnitude ~0.02. An AdamW update is ~`lr`
in size (the moment ratio is O(1)), i.e. `1e-4` in this campaign: **below half
an ULP for the large pretrained ViT weights, above it for the small random-init
ones**. The pretrained weights therefore get their updates rounded away every
step, forever — measured on the rd5/rd6 run, 63.6% of the pretrained ViT
weights were bit-for-bit unchanged between step 1000 and step 11326 (2.77%
drift) versus 19.5% drift in the random-init arm. Both the "pretrained visual
prior hurts" and the "post-bottleneck depth hurts" conclusions are confounded
with that artifact, and the confounder is collinear with the treatment.

`--fp32_master_weights` (env `FP32_MASTER_WEIGHTS=1`) makes AdamW keep an fp32
**master copy** of the selected bf16 parameters: the model stays bf16 for
forward/backward, the optimizer steps the fp32 masters with fp32 moments and
rounds back into the bf16 tensor. Sub-ULP updates now accumulate instead of
being destroyed.

`--fp32_master_scope` (env `FP32_MASTER_SCOPE`, `all` | `vision`, default
`all`) decides what it applies to. **Use `vision` for the controlled
experiment**:

| scope | mastered params | extra optimizer state | what changes |
|---|---|---|---|
| `all` | ~1.836B (everything trainable) | **+20.5 GiB** | decoder + text encoder + projectors + ViT numerics all at once |
| `vision` | ~86.4M (ViT/mixer only) | **+1.0 GiB** | ViT/mixer only; every other tensor stays on the bit-for-bit legacy path |

Note that **the decoder is trainable** in this codebase (`trainable params:
1835876608 || all params: 1874560000 || trainable%: 97.94`) — only the lower
text-encoder layers are frozen. `scope=all` is therefore expensive and
scientifically blunt; `scope=vision` is the isolated test of "ViT update
quantization explains the gap".

```bash
FP32_MASTER_WEIGHTS=1 FP32_MASTER_SCOPE=vision bash CVLM/scripts/run_full_pipeline.sh
# or directly:
python src/train_cvlm.py --output_dir "${OUTPUT_DIR}" \
  --fp32_master_weights true --fp32_master_scope vision ...
```

Ready-made two-arm control on the FULL original protocol (`randomvit` first —
it defends the central post-bottleneck-depth claim; `vit` second):
`hse/fp32ctrl.sbatch` — `sbatch --export=ALL,ARM=randomvit fp32ctrl.sbatch`.

Notes:
- `0` (default) = legacy in-place bf16 AdamW, byte-for-byte identical.
- Memory cost: +12 bytes per **mastered** parameter (fp32 master + fp32 grad
  buffer + fp32 instead of bf16 moments). Checkpoint dtype does not change;
  only `trainer_step_*.pt` grows (it now carries the masters).
- **Resume must reuse the same flag AND the same scope.** The masters live in
  the optimizer state; resuming a master-weights run without the flag raises
  instead of silently discarding the full-precision weights, and a scope change
  is rejected by the master-count check.
- fp32 parameters already in the model (`pool_queries`, `mixer_gates`) are
  optimized directly, with no master copy.

### ViT learnability diagnostic (always on)

Every `--log_interval` steps the trainer prints, and logs to trackio /
`mechanistic.jsonl`, the direct measurement of the problem above over the
trainable ViT (or mixer) stack:

```
    vit_frozen_frac=0.6357 vit_drift=0.0277 vit_master_drift=0.0312 vit_master_resid=0.001900 vit_master_resid_max=3.052e-03 (tracked=85056000 params, window=10 steps)
```

- `vit_frozen_frac` — fraction of individual weights whose **bit pattern** is
  unchanged since the previous log step (1.0 = the stack is not moving at all).
- `vit_drift` — relative L2 distance from initialization, `||W-W0||/||W0||`,
  computed on the **live bf16** weights (what the forward pass sees).
- `vit_master_drift` — the same drift on the **fp32 masters**: how far the
  optimizer has actually moved the stack. Only with `--fp32_master_weights`.
- `vit_master_resid`, `vit_master_resid_max` — `||master - live_bf16||`
  (relative to `||W0||`) and its max-abs: update that has been **accumulated
  but is not yet visible** in bf16 because it is still under half an ULP. The
  legacy path destroys exactly this quantity every step.

`W0` is read from `mechanistic_init.safetensors` when present, so drift stays
anchored to the true step-0 weights across resumes.

trackio keys: `train/vit_frozen_frac`, `train/vit_drift`,
`train/vit_master_drift`, `train/vit_master_resid`,
`train/vit_master_resid_max`.
