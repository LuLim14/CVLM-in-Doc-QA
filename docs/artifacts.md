# Artifacts and provenance

## Supplied publication files

- [paper.pdf](paper.pdf): the authors' final 16-page manuscript supplied for the release.
- [poster.pdf](poster.pdf): the presentation poster supplied for DocInsights 2026.
- [assets/architecture.png](assets/architecture.png): Figure 3 rendered from page 4 of that manuscript.
- [paper_metrics.json](../results/paper_metrics.json): aggregate table values transcribed from the manuscript. The file identifies the table, dataset slice, and metric for each group of results.

The paper is the source of numerical claims in the README. Reported values have not been recomputed during repository preparation. A file manifest records SHA-256 digests for the PDFs.

## Source code

The release starts from local snapshot `17223f6` and includes the subsequent local BM25 reader and LLMLingua threshold-selection changes. Model training uses the deterministic epoch sampler in this snapshot. The discovery run in the paper used an earlier sampler; seeds 43 and 44 are the paired replications under the deterministic sampler.

The code exposes additional experimental switches beyond the configurations reported in the paper. The [reproduction guide](reproduction.md) identifies the settings used for the main model. The `hse/` files are historical cluster recipes with environment-specific paths and run identifiers. Public instructions use direct Python entrypoints.

The available snapshot does not include the separate Qwen evaluation implementation used by the historical `CVLM_EVAL_FP16` cluster invocation. The main Qwen results remain available in the paper and the aggregate results file. The generic model-name option alone is not a recovered exact recipe for that numerical path.

## Availability

| Artifact | Included |
|:--|:--|
| Model, training, evaluation, and selector source | Yes |
| CPU tests and optional CUDA checks | Yes |
| Final paper and poster | Yes |
| Aggregate values from the paper | Yes |
| Historical trained weights | No |
| Optimizer checkpoints and full run logs | No |
| Original per-example predictions and evaluation dumps | No |
| Original judge annotations and representation-probe outputs | No |
| Third-party source documents | No |

New training runs save checkpoints and architecture manifests. Evaluation can produce aggregate JSON and per-example JSONL. The latter may contain source text, reference answers, and generated answers, and is ignored by Git by default. Download PwC and upstream model weights from their original providers under their respective terms.

## Reading the results

PPL is the exponential of answer-token-weighted mean NLL. Token accuracy is total correct answer tokens divided by total answer tokens. Paired headline intervals resample whole documents, keeping all questions for each sampled document together.

The nominal `cr` uses the encoder tokenizer. The comparison controls the cap on document positions in the decoder context. Selectors can use fewer than the cap. The frozen full-document reference is listed separately from the trained common-cap systems.

The depth-replication checkpoints and the additional headline system-replication checkpoints are different runs. They must not be pooled into one seed table. First-sentence generation is reported separately from teacher-forced metrics because the compressor and SFT training paths differ in EOS supervision.

System timings are the specific A100 microbenchmarks described in the paper. Non-decoder parameter counts exclude the language decoder, while timing and memory include it.

## Release checks

On 29 October 2026, a fresh Python 3.11.15 environment installed `requirements.txt` and `requirements-dev.txt` successfully. The offline CPU suite passed with **285 passed, 2 skipped**, using PyTorch 2.5.1 and Transformers 4.57.6 on macOS ARM64. The skipped checks require CUDA/Liger. The Linux dependency set, including the optional GPU and selector packages, also resolved successfully. Full GPU training was not rerun during packaging.

The documented CLI options were checked against the actual parsers. Local documentation links and PDF hashes were verified, and `CITATION.cff` was validated against the CFF 1.2.0 schema. The included GitHub Actions workflow runs the CPU suite on Linux when enabled in the repository.
