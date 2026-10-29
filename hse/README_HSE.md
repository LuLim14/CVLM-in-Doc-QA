# Historical cluster recipes

These scripts record the Slurm campaigns used during the experiments. They contain the original cluster's partition names, module environment, absolute paths, snapshot identifiers, and checkpoint locations. They are retained as protocol records and require adaptation before use on another system.

Use the [reproduction guide](../docs/reproduction.md) for portable training and evaluation commands. The public entrypoints are `src/train_cvlm.py`, `src/train_sft.py`, and `src/eval_cvlm.py`.

`protocol.env` records the main curriculum, source and context limits, optimizer settings, and model identifiers. It also sets offline cache paths and evaluation defaults specific to the cluster. In particular, inherited `EVAL_CR_SCHEDULE` and `EVAL_ALL_CHECKPOINTS` values are incompatible with some isolated reader evaluations. The reproduction guide clears them explicitly.

The recipe names describe historical campaigns. Some exploratory arms were not part of the final paper. The final [paper](../docs/paper.pdf) and [reported aggregates](../results/paper_metrics.json) identify the published comparisons.

The launchers refer to run directories and immutable snapshots that are not distributed here. Several scripts manage or remove old training checkpoints inside their configured output directories. Read and adapt the paths and retention settings before submitting any job.
