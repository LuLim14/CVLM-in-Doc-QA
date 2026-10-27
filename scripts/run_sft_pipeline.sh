#!/usr/bin/env bash
# SFT baseline pipeline: train -> eval.
#
# Trains the same decoder used by CVLM on (document + question -> answer) with
# standard supervised fine-tuning (cross-entropy on answer tokens, prompt + pad
# masked to -100). After training, runs eval_cvlm.py --mode sft against the
# checkpoint to produce a JSON shape-compatible with CVLM's eval JSON.
#
# Typical use:
#   tmux new -s sft
#   OUTPUT_DIR=/path/to/sft_run bash CVLM/scripts/run_sft_pipeline.sh
#   <Ctrl-b d>
#   tmux attach -t sft

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Ignore ~/.local site-packages so the conda env's torch wins (avoids
# `libtorch_cuda.so: undefined symbol: ncclDevCommDestroy` from a user-site
# torch built against a newer NCCL than the system's).
export PYTHONNOUSERSITE=1
# Flush print() to tee/pipeline.log immediately instead of waiting for the
# 8KB block-buffer to fill — surfaces master-rank prints + any TrackioRun
# warnings in real time.
export PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# Configuration (override by exporting env vars before running).
# -----------------------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts/MoE-losses/cvlm/sft_run_$(date +%Y%m%d_%H%M%S)}"
DATASET_NAME="${DATASET_NAME:-sggetao/PwC}"

# Model
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolLM2-1.7B-Instruct}"
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-answerdotai/ModernBERT-base}"

# Train
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-5}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_ANSWER_LEN="${MAX_ANSWER_LEN:-1024}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL_STEPS="${SAVE_INTERVAL_STEPS:-500}"
PLOT_INTERVAL="${PLOT_INTERVAL:-100}"
NPROC="${NPROC:-1}"
ENABLE_WARMUP="${ENABLE_WARMUP:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

# Eval — these mirror CVLM's eval so the JSONs are diffable.
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
EVAL_COMPUTE_GEN="${EVAL_COMPUTE_GEN:-1}"
NUM_CACHED_SAMPLES="${NUM_CACHED_SAMPLES:-8}"   # N decoded {doc,q,ref,pred} tuples dumped next to eval JSON; 0 disables
# These are needed only so CvlmTrainDataset constructs the same filtered set
# the CVLM eval saw. The SFT decoder doesn't consume vision tokens.
MAX_VISION_LEN="${MAX_VISION_LEN:-256}"
MAX_SOURCE_LEN="${MAX_SOURCE_LEN:-0}"
COMPRESSION_RATE="${COMPRESSION_RATE:-4}"

# trackio (replaces TensorBoard for the live UI)
TRACKIO_PROJECT="${TRACKIO_PROJECT:-cvlm}"
TRACKIO_RUN_NAME="${TRACKIO_RUN_NAME:-$(basename "${OUTPUT_DIR}")_bs${BATCH_SIZE}_lr${LR}_ep${EPOCHS}_pl${MAX_PROMPT_LEN}_al${MAX_ANSWER_LEN}}"
TRACKIO_DISABLE="${TRACKIO_DISABLE:-0}"
export TRACKIO_PROJECT TRACKIO_RUN_NAME TRACKIO_DISABLE
# Only export TRACKIO_SPACE_ID when set non-empty: trackio reads the env var
# directly and treats an empty string as "<user>/" which fails repo-id
# validation.
if [[ -n "${TRACKIO_SPACE_ID:-}" ]]; then
  export TRACKIO_SPACE_ID
else
  unset TRACKIO_SPACE_ID
fi

TB_DIR="${OUTPUT_DIR}/tb"
LOG_FILE="${OUTPUT_DIR}/pipeline.log"
mkdir -p "${OUTPUT_DIR}" "${TB_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================================================"
echo "SFT BASELINE PIPELINE"
echo "  OUTPUT_DIR        = ${OUTPUT_DIR}"
echo "  DATASET_NAME      = ${DATASET_NAME}"
echo "  MODEL_NAME        = ${MODEL_NAME}"
echo "  TEXT_ENCODER_NAME = ${TEXT_ENCODER_NAME}  (eval-time filter only)"
echo "  EPOCHS            = ${EPOCHS}"
echo "  BATCH_SIZE        = ${BATCH_SIZE}"
echo "  LR                = ${LR}"
echo "  MAX_PROMPT_LEN    = ${MAX_PROMPT_LEN}"
echo "  MAX_ANSWER_LEN    = ${MAX_ANSWER_LEN}"
echo "  MAX_SAMPLES       = ${MAX_SAMPLES}"
echo "  PLOT_INTERVAL     = ${PLOT_INTERVAL}"
echo "  NPROC             = ${NPROC}"
echo "  EVAL_SPLIT        = ${EVAL_SPLIT}"
echo "  NUM_CACHED_SAMPLES = ${NUM_CACHED_SAMPLES}"
echo "  TB_DIR            = ${TB_DIR}"
echo "  LOG_FILE          = ${LOG_FILE}"
echo "  TRACKIO_PROJECT   = ${TRACKIO_PROJECT}"
echo "  TRACKIO_RUN_NAME  = ${TRACKIO_RUN_NAME}"
echo "  TRACKIO_SPACE_ID  = ${TRACKIO_SPACE_ID:-<local-only>}"
echo "  TRACKIO_DISABLE   = ${TRACKIO_DISABLE}"
echo "======================================================================"
echo "Live PNGs:    ls ${OUTPUT_DIR}/*.png   (loss.png, lr.png, grad_norm.png,"
echo "              batch_time.png, dashboard.png — refreshed every"
echo "              ${PLOT_INTERVAL} optimizer steps)"
echo "trackio UI:   trackio show --project ${TRACKIO_PROJECT}"
echo "(or set TRACKIO_SPACE_ID=user/space to host on HF Spaces)"
echo "======================================================================"

# -----------------------------------------------------------------------------
# Step 1/2: SFT train (skip with SKIP_TRAIN=1)
# -----------------------------------------------------------------------------
SKIP_TRAIN="${SKIP_TRAIN:-0}"
if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo; echo "===== Step 1/2: SKIPPED (SKIP_TRAIN=1) ====="
else
  echo; echo "===== Step 1/2: SFT train -> ${OUTPUT_DIR} ====="
  TRAIN_ARGS=(
    "${ROOT}/src/train_sft.py"
    --output_dir "${OUTPUT_DIR}"
    --dataset_name "${DATASET_NAME}"
    --model_name_or_path "${MODEL_NAME}"
    --max_samples "${MAX_SAMPLES}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --max_prompt_len "${MAX_PROMPT_LEN}"
    --max_answer_len "${MAX_ANSWER_LEN}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --log_interval "${LOG_INTERVAL}"
    --save_interval_steps "${SAVE_INTERVAL_STEPS}"
    --plot_interval "${PLOT_INTERVAL}"
    --tensorboard_dir "${TB_DIR}/train"
  )
  if [[ "${ENABLE_WARMUP}" == "1" ]]; then
    TRAIN_ARGS+=( --enable_warmup --warmup_steps "${WARMUP_STEPS}" )
  fi
  if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
    TRAIN_ARGS+=( --gradient_checkpointing )
  fi
  if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC}" "${TRAIN_ARGS[@]}"
  else
    python "${TRAIN_ARGS[@]}"
  fi
fi

# -----------------------------------------------------------------------------
# Step 2/2: Eval the SFT checkpoint
# -----------------------------------------------------------------------------
echo; echo "===== Step 2/2: Eval SFT checkpoint (mode=sft) ====="
EVAL_CMD=(
  python "${ROOT}/src/eval_cvlm.py"
  --mode sft
  --sft_model_path "${OUTPUT_DIR}"
  --dataset_name "${DATASET_NAME}"
  --dataset_split "${EVAL_SPLIT}"
  --model_name_or_path "${MODEL_NAME}"
  --text_encoder_name "${TEXT_ENCODER_NAME}"
  --compression_rate "${COMPRESSION_RATE}"
  --max_samples "${EVAL_MAX_SAMPLES}"
  --batch_size "${EVAL_BATCH_SIZE}"
  --max_prompt_len "${MAX_PROMPT_LEN}"
  --max_answer_len "${MAX_ANSWER_LEN}"
  --max_vision_len "${MAX_VISION_LEN}"
  --max_source_len "${MAX_SOURCE_LEN}"
  --tensorboard_dir "${TB_DIR}"
  --tb_run_name "eval_sft"
  --output_json "${OUTPUT_DIR}/eval_sft.json"
  --num_cached_samples "${NUM_CACHED_SAMPLES}"
)
if [[ "${EVAL_COMPUTE_GEN}" == "1" ]]; then
  EVAL_CMD+=( --compute_generation_metrics )
fi
"${EVAL_CMD[@]}"

echo
echo "======================================================================"
echo "DONE. Artifacts:"
echo "  HF model dir : ${OUTPUT_DIR}"
echo "  eval JSON    : ${OUTPUT_DIR}/eval_sft.json"
echo "  PNG plots    : ${OUTPUT_DIR}/dashboard.png (+ loss/lr/grad_norm/batch_time)"
echo "  metrics CSV  : ${OUTPUT_DIR}/metrics.csv"
echo "  full log     : ${LOG_FILE}"
echo "  trackio UI   : trackio show --project ${TRACKIO_PROJECT}"
echo "======================================================================"
