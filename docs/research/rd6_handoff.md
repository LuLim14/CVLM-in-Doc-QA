# RD6 handoff: positional holes, method gate, and mechanistic artifacts

## What is established

The direct same-checkpoint mask intervention changed held-out PPL by at most
about 0.5%, far below the observed 10--177% depth penalty. ViT PAD-attention is
therefore rejected as the explanation of the `no-ViT < random-ViT` result. The
mask remains a correctness fix; do not spend three GPU-days on the
masked/unmasked training campaign for this claim.

The synthetic-padding diagnostic exposed a different issue. CVLM teacher
forcing previously let the decoder infer absolute positions `0..L-1`, so
masked latent holes between the document and answer shifted answer positions.
Hugging Face generation already constructs positions as
`attention_mask.cumsum(-1)-1`; training and teacher-forcing eval therefore used
different positional semantics. Synthetic PAD cannot truncate real latents in
RD5/RD6: it is appended after pooling and capped by the remaining
`max_vision_len` capacity.

## Run order

1. Deploy one immutable snapshot (the new scripts default to
   `/home/aburlova/cvlm_rd6`; override with `CVLM_SNAPSHOT_ROOT`).
2. Run the eval-only compact-position diagnosis on the existing no-ViT
   checkpoint:

   ```bash
   sbatch --export=ALL,CVLM_SNAPSHOT_ROOT=/home/aburlova/cvlm_rd6 \
     hse/position_diag.sbatch
   ```

3. Compare `compact=false` versus `compact=true`, especially
   `batch_size_ppl_delta`, `bucketing_ppl_delta`, and the synthetic-padding
   curves at CR=1. This changes no learned weights.
4. Do not launch `maskcmp.sbatch` or `ratedepth.sbatch` while this method gate
   is unresolved.

## Decision rule for the method

- If compact positions remove the no-ViT synthetic-padding degradation and
  batch-composition dependence, the method candidate is
  **position-safe latent packing**: internal padding requires both an
  attention mask and mask-compacted decoder position IDs. The next experiment
  is then a paired legacy-vs-compact no-ViT retrain, not a masked ViT campaign.
- If compact positions do not remove the effect, inspect
  `synthetic_slots_per_sample` and per-sample dumps before assigning a
  mechanism. Do not describe the effect as truncation: the intervention does
  not discard pooled content.
- ViT attention mass is stored in `padding_attention_mass_by_layer`.
  `attention_batches` is only the configured sampling budget.

## Artifacts saved without another training run

Each new run writes:

- `initialization_manifest.json`: architecture, package/model revisions, exact
  per-module and per-layer initialization hashes;
- `mechanistic_init.safetensors`: exact step-0 compression-path weights for
  later layerwise weight-drift analysis;
- `mechanistic_probe.pt`: fixed tokenized examples for checkpoint-level
  activation/CKA probes without reloading a mutable dataset;
- `mechanistic.jsonl`: CR, batch hash, source/latent lengths, padding fraction,
  compact-position mode, gates, gate gradients, valid-token contribution
  ratios, and per-layer post-bottleneck gradient norms;
- `model_step_*.safetensors`: retained at every configured checkpoint interval;
- `CODE_SHA256`, `CODE_VERSION`, and `EXPERIMENT_FLAGS`.

Resume verifies initialization, dependency versions, architecture and the
cached probe. Eval rejects checkpoint/architecture mismatches that would
otherwise be hidden by `strict=False`.

## Validation

```bash
python -m pytest -q
bash -n hse/position_diag.sbatch hse/padding_diag.sbatch hse/maskcmp.sbatch
```

The compact-position unit test directly verifies that a tiny Llama decoder's
answer states are invariant to inserted masked holes only when compact
position IDs are supplied.

Current local result: 137 passed, 2 skipped.
