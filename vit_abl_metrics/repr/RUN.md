# Representation analysis — run command

Runs the hidden-states mechanistic analysis (CKA / transform-magnitude / weight-drift /
effective-rank) across the three ViT-ablation arms at cr=16.

```bash
cd /home/jovyan/shares/SR008.fs2/acherepanov/compress_project_new/CVLM
PYTHONNOUSERSITE=1 TRACKIO_DISABLE=1 \
  /home/jovyan/.mlspace/envs/cvlm/bin/python vit_abl_metrics/repr_analysis.py \
  2>&1 | tee logs/repr_analysis.log
```

Outputs -> `vit_abl_metrics/repr/`:
- `repr_metrics.json`, `repr_metrics.csv`
- `transform_magnitude.png`, `cosine_alignment.png`, `cross_arm_cosine.png`,
  `effective_rank.png`, `weight_drift.png`
- `cka_vit_output.png`, `cka_vision_embeds.png`

`cosine_alignment.png`: per-token cosine(vit_input, vit_output) — scale-invariant angular
reorientation. cos==1 => ViT only rescales/passes direction through; low cos => genuine
reorientation. `none` arm is 1.0 by construction (bypass).

`cross_arm_cosine.png`: per-token cosine between two arms' vit_output on ALIGNED rows
(same doc+position; valid because extract() uses shuffle=False + identical cr=16 geometry).
Raw-direction counterpart to CKA: CKA is rotation-invariant, this asks whether the arms
literally point the same way per token. Reported for arm pairs vit/random_vit, vit/none,
random_vit/none.

## Known-fixed issue
`weight_drift` for random_vit previously exploded (values ~1e11) because relative drift
‖w1-w0‖/‖w0‖ divides by ~0 for zero-initialised biases/LayerNorm. Now tensors with
init-norm < 1e-6 are skipped, so only the (nonzero-init) weight matrices are reported.

## Note before running (the "deadlock")
The last attempt was NOT a logic deadlock — the `vit` arm's model build triggered a
**network download of SmolLM2-1.7B-Instruct** (not in this server's HF cache) under
rate-limited unauthenticated HF, so it blocked on network I/O before printing any arm.
Pre-warm the cache or set a token:

```bash
export HF_TOKEN=<your token>          # avoids the slow unauthenticated download
# or pre-download once:
PYTHONNOUSERSITE=1 /home/jovyan/.mlspace/envs/cvlm/bin/huggingface-cli \
  download HuggingFaceTB/SmolLM2-1.7B-Instruct
```

ModernBERT-base and google/vit-base-patch16-224 are also required in cache.

## Config baked into the script (edit at top of repr_analysis.py if needed)
- CKPT_STEP = 37751 (final cr=16 checkpoint), CR = 16
- N_DOCS = 256 shared PwC test docs, BATCH_SIZE = 4
- arms: vit / random_vit / none (final `model_step_37751.safetensors`)
