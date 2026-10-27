# tmux workflow (survives terminal disconnect)
Do you need tmux? Strictly speaking, no — nohup ... & also works — but tmux is what you actually want because:

You can reattach and watch live logs interactively.
It survives SSH drops, laptop sleep, and intentional disconnects.
No special handling of stdout/stderr — the pipeline script already tees to pipeline.log, but tmux keeps the live console too.
One-liner you can copy-paste

# Create a detached session and launch the pipeline inside it
Last version
```
conda activate cvlm 
tmux new -s cvlm -d "bash /home/jovyan/shares/SR008.fs2/acherepanov/compress_project_new/CVLM/scripts/run_full_pipeline.sh; exec bash"
```

```
tmux new -s cvlm -d "conda activate cvlm && \
  OUTPUT_DIR=$HOME/cvlm_runs/run1 \
  EMBEDDINGS_PATH=$HOME/cvlm_runs/emb.npz \
  MAX_VISION_LEN=256 MAX_ANSWER_LEN=256 MAX_PROMPT_LEN=512 \
  COMPRESSION_RATE=4 EPOCHS=1 BATCH_SIZE=2 \
  bash /home/jovyan/shares/SR008.fs2/acherepanov/compress_project_new/CVLM/scripts/run_full_pipeline.sh"
```
Then disconnect/close the terminal whenever you want. To come back:

```
tmux ls                      # see the running session
tmux attach -t cvlm          # reattach (Ctrl-b d to detach again)
tail -f $HOME/cvlm_runs/run1/pipeline.log   # or just follow the log from any shell
```

# Watch TensorBoard while it runs
The pipeline logs both train and eval under $OUTPUT_DIR/tb:

`…/tb/train/…` ← training scalars (loss, lr, grad_norm, batch_time)

`…/tb/eval_cvlm/…` ← eval_cvlm scalars + compression_ratio histogram

`…/tb/eval_baseline_llm/…` ← baseline for comparison

Start TB in a second tmux window (or any shell):

```
tmux new -s tb -d "conda activate cvlm && \
  tensorboard --logdir $HOME/cvlm_runs/run1/tb --port 6006 --bind_all"
```

Then open `http://<host>:6006` in your browser. You'll see training loss over steps and eval metrics plotted at the checkpoint's step index (that's what `--global_step ${STEP}` does — eval scalars are stamped with the training step they were computed from, so they line up on the same x-axis as train curves).

# Minimal alternative without tmux
If tmux isn't available for some reason:

```
nohup bash CVLM/scripts/run_full_pipeline.sh > /dev/null 2>&1 &
disown
tail -f "${OUTPUT_DIR}/pipeline.log"
```

This is fire-and-forget; you won't be able to reattach to the live console, but the log file has everything. Not recommended vs. tmux, but it works.

# Knobs for the pipeline script
All configurable via env vars before invoking run_full_pipeline.sh:

| Var | Default | What it controls |
|---|---|---|
| `OUTPUT_DIR` | `runs/cvlm_<timestamp>` |	Root of all run artifacts |
| `EMBEDDINGS_PATH` |	`data/processed/embeddings.npz` |	Where preprocess writes / train+eval read |
| `SKIP_PREPROCESS` |	`0`	| Set to 1 to reuse an existing .npz |
| `COMPRESSION_RATE` |	`4` | Chunk size for token-level mean pooling in preprocess |
| `MAX_VISION_LEN / MAX_PROMPT_LEN / MAX_ANSWER_LEN` |	256 / 512 / 256	| Dataset length caps — must be identical at train and eval time (you've already hit this once) |
| `EPOCHS / BATCH_SIZE / LR / GRAD_ACCUM` | 1 / 2 / 1e-5 / 1 | Trainer basics |
| `NPROC` |	`1` | `>1` runs via `torchrun` for DDP |
| `EVAL_MODES` |	`"cvlm baseline_llm"` |	Space-separated list of modes; comment baseline_proj back in if you want the random-projection baseline too |
| `EVAL_MAX_SAMPLES` |	`0` (all) |	Cap eval sample count for fast smoke tests |
Example: a 256-sample smoke run you can fire in one shot:

```
tmux new -s cvlm-smoke -d "conda activate cvlm && \
  OUTPUT_DIR=/tmp/cvlm_smoke \
  EMBEDDINGS_PATH=/tmp/cvlm_smoke/emb.npz \
  EMBED_MAX_SAMPLES=256 EVAL_MAX_SAMPLES=64 \
  MAX_VISION_LEN=256 EPOCHS=1 BATCH_SIZE=2 \
  bash CVLM/scripts/run_full_pipeline.sh"
```
Then ```tensorboard --logdir /tmp/cvlm_smoke/tb --port 6006 --bind_all``` to view the whole train+eval trace in one place.