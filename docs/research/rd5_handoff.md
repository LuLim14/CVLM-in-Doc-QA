# RD5 handoff: padding confound, method gate, and mechanistic artifacts

## What is established

The old `no-ViT < random-ViT < pretrained-ViT` training-loss table is not yet
an architectural result. In the legacy path, post-bottleneck ViT blocks could
attend to padded latent slots, while the no-ViT branch only exposed those slots
to the decoder attention mask. The falling depth penalty at larger CR is
therefore also predicted by a padding-contamination hypothesis.

The repository now has a key-padding-safe ViT/mixer path and a direct
masked/unmasked control. Do not restart the rate-depth campaign until the
padding diagnosis is read.

## Run order

1. Deploy one immutable snapshot (the scripts default to
   `/home/aburlova/cvlm_rd5`; override with `CVLM_SNAPSHOT_ROOT`).
2. Run the eval-only diagnosis on existing, common-step checkpoints:

   ```bash
   sbatch --export=ALL,CVLM_SNAPSHOT_ROOT=/home/aburlova/cvlm_rd5 \
     hse/padding_diag.sbatch
   ```

3. Inspect the JSON results before allocating three full training jobs.
4. If padding materially moves held-out PPL, launch the controlled campaign
   with one shared `MC_STAMP`:

   ```bash
   for arm in novit unmasked masked; do
     sbatch --job-name="mask_${arm}" \
       --export=ALL,ARM="${arm}",MC_STAMP=rd5,CVLM_SNAPSHOT_ROOT=/home/aburlova/cvlm_rd5 \
       hse/maskcmp.sbatch
   done
   ```

The training arms share initialization, sampler order, curriculum and
post-bottleneck blocks. `unmasked` versus `masked` differs only in the key
padding mask. The final ViT LayerNorm is disabled in both direct-stack arms so
the comparison against no-ViT does not silently include an extra normalization
operation. Final checkpoints are evaluated at CR 1/2/4/8/16/32; generation is
enabled at CR=16 by default.

## Decision rule for the method

- If synthetic padding and batch composition change legacy PPL, attention puts
  non-trivial mass on PAD, and the same checkpoint becomes invariant under
  mask-safe inference, the original depth table was confounded. The full
  masked/unmasked retrain then tests a publishable method claim:
  **padding-invariant post-bottleneck latent mixing**.
- If the interventions do not move PPL, padding is ruled out as the explanation
  and a mask alone is not a method contribution. Keep it as a correctness fix;
  do not relabel the negative rate-depth result.
- Attention mass is mechanistic support, not the causal endpoint. The causal
  outcomes are paired PPL changes under synthetic padding/mask intervention
  and the masked/unmasked retrain.

## Artifacts saved without another training run

Each new run writes:

- `initialization_manifest.json`: architecture, package/model revisions, exact
  per-module and per-layer initialization hashes;
- `mechanistic_init.safetensors`: exact step-0 compression-path weights for
  later layerwise weight-drift analysis;
- `mechanistic_probe.pt`: fixed tokenized examples for checkpoint-level
  activation/CKA probes without reloading a mutable dataset;
- `mechanistic.jsonl`: CR, batch hash, source/latent lengths, padding fraction,
  gates, gate gradients, valid-token contribution ratios, and per-layer
  post-bottleneck gradient norms;
- `model_step_*.safetensors`: retained at every configured checkpoint interval;
- `CODE_SHA256`, `CODE_VERSION`, and `EXPERIMENT_FLAGS`.

Resume verifies initialization, dependency versions, architecture and the
cached probe. Eval rejects checkpoint/architecture mismatches that would
otherwise be hidden by `strict=False`.

## Validation

```bash
python -m pytest -q
bash -n hse/padding_diag.sbatch hse/maskcmp.sbatch hse/ratedepth.sbatch
```

Current local result: 135 passed, 2 skipped.
