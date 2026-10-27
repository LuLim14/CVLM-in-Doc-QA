#!/usr/bin/env bash
# Full CVLM pipeline: train -> eval.
#
# On-the-fly text-encoder variant: ModernBERT lives inside the CVLM model and
# runs during the forward pass, so there is no preprocessing step. Both stages
# log into the same TensorBoard tree at "${OUTPUT_DIR}/tb", so a single
# `tensorboard --logdir ${OUTPUT_DIR}/tb` shows train + eval.
#
# Designed to be safe under terminal disconnect: stdout/stderr are tee'd into
# "${OUTPUT_DIR}/pipeline.log" and the script exits non-zero on any failure.
#
# Typical use:
#   tmux new -s cvlm
#   OUTPUT_DIR=/path/to/run bash CVLM/scripts/run_full_pipeline.sh
#   <Ctrl-b d>   # detach — training keeps running
#   tmux attach -t cvlm   # reattach later

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
# Configuration (override by exporting env vars before running the script).
# -----------------------------------------------------------------------------
DATASET_NAME="${DATASET_NAME:-sggetao/PwC}"

# Model
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolLM2-1.7B-Instruct}"
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-answerdotai/ModernBERT-base}"
# Must be loadable as transformers.ViTModel (not CLIP/SigLIP/DINO without code changes).
VISION_ENCODER_NAME="${VISION_ENCODER_NAME:-google/vit-base-patch16-224}"
# VISION_ENCODER_NAME="${VISION_ENCODER_NAME:-google/vit-large-patch16-224}" # Lets try with another vision encoder

# Train
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_ANSWER_LEN="${MAX_ANSWER_LEN:-1024}"
MAX_VISION_LEN="${MAX_VISION_LEN:-1024}"
MAX_SOURCE_LEN="${MAX_SOURCE_LEN:-0}"      # 0 = compression_rate * max_vision_len
COMPRESSION_RATE="${COMPRESSION_RATE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"            # cap HF dataset rows; 0 = all
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL_STEPS="${SAVE_INTERVAL_STEPS:-0}"
NPROC="${NPROC:-1}"
# Linear warmup stabilises the first ~100 optimizer steps. With the projectors
# + ViT freshly initialised, early grad norms can spike above 100; without
# warmup clip=1.0 silently discards most of those updates.
ENABLE_WARMUP="${ENABLE_WARMUP:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
UNFREEZE_ENCODER_TOP_K="${UNFREEZE_ENCODER_TOP_K:-22}"   # 0 = encoder fully frozen; >0 unfreezes top-K layers + final_norm, 22 == fully unreeze for ModerBert
NUM_POOL_LATENTS="${NUM_POOL_LATENTS:-4}"                # K learnable queries per chunk for the attention pool; K>1 = Perceiver-style multi-query pool whose K outputs are mean-pooled to 1 vision token per chunk

# Vision-stage ablation (must match at train and eval). One run per backend —
# launch each with its own OUTPUT_DIR.
#   vit        = pretrained ViT (default / base setup for comparison)
#   random_vit = same architecture, random weights (isolates pretrained prior)
#   none       = bypass the ViT transformer+layernorm (isolates ViT computation)
VISION_BACKEND="${VISION_BACKEND:-vit}"

# 1 = plain projectors (no input_norm.* in safetensors — old runs before RMSNorm)
LEGACY_PROJECTOR="${LEGACY_PROJECTOR:-0}"
export LEGACY_PROJECTOR

CR_SCHEDULE="${CR_SCHEDULE:-1:6000,2:12000,4:18000,8:24000,16:0}"   # empty = static cr from COMPRESSION_RATE
DISABLE_LR_SCHEDULE="${DISABLE_LR_SCHEDULE:-1}"   # 1 = constant LR after warmup (no cosine decay)
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"   # 1 = wrap doc+question+answer in the decoder chat template (train + eval must match)

# Output dir — default name embeds main hyperparameters for easy A/B comparison.
# Override by exporting OUTPUT_DIR=/abs/path before running the script.
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts/MoE-losses/cvlm/run_$(date +%Y%m%d_%H%M%S)_layer_norm_cr${COMPRESSION_RATE}_vl${MAX_VISION_LEN}_sl${MAX_SOURCE_LEN}_bs${BATCH_SIZE}_lr${LR}_ep${EPOCHS}_gc${GRADIENT_CHECKPOINTING}_unfrzed${UNFREEZE_ENCODER_TOP_K}_lt${NUM_POOL_LATENTS}}"
# OUTPUT_DIR=/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts/MoE-losses/cvlm/run_20260515_231650_vit_large_cr1_vl1024_sl0_bs64_lr1e-4_ep10_gc1_unfrzed22_lt4

# Eval
EVAL_SPLIT="${EVAL_SPLIT:-test}"          # use 'test' once you have one
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
# Default eval suite covers: cvlm (method), cvlm_shuffle (source-relevance sanity
# check), baseline_llm (no-document lower bound), baseline_llm_full (oracle
# upper bound with full document), baseline_proj (random-projection ablation).
EVAL_MODES="${EVAL_MODES:-cvlm baseline_llm baseline_llm_full}"
EVAL_COMPUTE_GEN="${EVAL_COMPUTE_GEN:-1}"  # 1 = run ROUGE / BLEU generation metrics
NUM_CACHED_SAMPLES="${NUM_CACHED_SAMPLES:-16}"   # N decoded {doc,q,ref,pred} tuples dumped per ckpt next to eval JSON; 0 disables
EVAL_ALL_CHECKPOINTS="${EVAL_ALL_CHECKPOINTS:-1}"  # 1 = sweep every model_step_*.safetensors
EVAL_CR_SCHEDULE="${EVAL_CR_SCHEDULE:-1:6000,2:12000,4:18000,8:24000,16:0}"           # if set + EVAL_ALL_CHECKPOINTS=1, eval each ckpt at the cr that trained it

# --- LongBench QA eval (eval-only; official leaderboard qa_f1_score) ---
# Runs after the PwC eval. Loops the QA subtasks x EVAL_MODES against
# THUDM/LongBench (test split), applying the official prompt template and
# per-task max_new_tokens. No sample is dropped: LB_MAX_SOURCE_LEN must be large
# enough to hold the longest document (documents are truncated to it only at the
# encoder). Set LONGBENCH_EVAL=0 to skip.
LONGBENCH_EVAL="${LONGBENCH_EVAL:-1}"
LONGBENCH_DATASET="${LONGBENCH_DATASET:-THUDM/LongBench}"
LONGBENCH_SPLIT="${LONGBENCH_SPLIT:-test}"
LONGBENCH_TASKS="${LONGBENCH_TASKS:-narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique}"
LB_MAX_SOURCE_LEN="${LB_MAX_SOURCE_LEN:-32768}"   # encoder-side truncation cap; keep every eval sample
LB_MAX_PROMPT_LEN="${LB_MAX_PROMPT_LEN:-2048}"    # template wrapper + question fit uncompressed

# trackio (replaces TensorBoard for the live UI)
TRACKIO_PROJECT="${TRACKIO_PROJECT:-cvlm}"
TRACKIO_RUN_NAME="${TRACKIO_RUN_NAME:-$(basename "${OUTPUT_DIR}")}"
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
# Log directory is overridable (LOG_DIR). Default keeps the log next to the
# checkpoints as pipeline.log (unchanged). When LOG_DIR is set to a shared dir,
# the file is prefixed with the run name so parallel ablations don't clobber
# each other's logs.
if [[ -n "${LOG_DIR:-}" && "${LOG_DIR}" != "${OUTPUT_DIR}" ]]; then
  LOG_FILE="${LOG_DIR}/$(basename "${OUTPUT_DIR}").pipeline.log"
else
  LOG_DIR="${OUTPUT_DIR}"
  LOG_FILE="${OUTPUT_DIR}/pipeline.log"
fi

mkdir -p "${OUTPUT_DIR}" "${TB_DIR}" "${LOG_DIR}"

# Tee everything to a log file so you can `tail -f` it from another shell.
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================================================"
echo "CVLM FULL PIPELINE (on-the-fly encoder)"
echo "  OUTPUT_DIR        = ${OUTPUT_DIR}"
echo "  DATASET_NAME      = ${DATASET_NAME}"
echo "  MODEL_NAME        = ${MODEL_NAME}"
echo "  TEXT_ENCODER_NAME = ${TEXT_ENCODER_NAME}"
echo "  VISION_ENCODER_NAME = ${VISION_ENCODER_NAME}"
echo "  MAX_VISION_LEN    = ${MAX_VISION_LEN}"
echo "  MAX_SOURCE_LEN    = ${MAX_SOURCE_LEN} (0 = cr*max_vision_len)"
echo "  COMPRESSION_RATE  = ${COMPRESSION_RATE}"
echo "  MAX_SAMPLES       = ${MAX_SAMPLES} (0 = all)"
echo "  ENABLE_WARMUP     = ${ENABLE_WARMUP} (steps=${WARMUP_STEPS})"
echo "  GRADIENT_CHECKPOINTING = ${GRADIENT_CHECKPOINTING}"
echo "  UNFREEZE_ENCODER_TOP_K = ${UNFREEZE_ENCODER_TOP_K}"
echo "  NUM_POOL_LATENTS  = ${NUM_POOL_LATENTS}"
echo "  VISION_BACKEND    = ${VISION_BACKEND}"
echo "  LEGACY_PROJECTOR  = ${LEGACY_PROJECTOR}"
echo "  CR_SCHEDULE       = ${CR_SCHEDULE:-<static>}"
echo "  DISABLE_LR_SCHEDULE = ${DISABLE_LR_SCHEDULE}"
echo "  USE_CHAT_TEMPLATE = ${USE_CHAT_TEMPLATE}"
echo "  EVAL_MODES        = ${EVAL_MODES}"
echo "  EVAL_COMPUTE_GEN  = ${EVAL_COMPUTE_GEN}"
echo "  NUM_CACHED_SAMPLES   = ${NUM_CACHED_SAMPLES}"
echo "  EVAL_ALL_CHECKPOINTS = ${EVAL_ALL_CHECKPOINTS}"
echo "  EVAL_CR_SCHEDULE     = ${EVAL_CR_SCHEDULE:-<static>}"
echo "  TB_DIR            = ${TB_DIR}"
echo "  LOG_FILE          = ${LOG_FILE}"
echo "  TRACKIO_PROJECT   = ${TRACKIO_PROJECT}"
echo "  TRACKIO_RUN_NAME  = ${TRACKIO_RUN_NAME}"
echo "  TRACKIO_SPACE_ID  = ${TRACKIO_SPACE_ID:-<local-only>}"
echo "  TRACKIO_DISABLE   = ${TRACKIO_DISABLE}"
echo "  SKIP_TRAIN        = ${SKIP_TRAIN:-0}  (set EVAL_ONLY=1 or SKIP_TRAIN=1 for eval-only)"
echo "======================================================================"
echo "trackio UI:   trackio show --project ${TRACKIO_PROJECT}"
echo "(or set TRACKIO_SPACE_ID=user/space to host on HF Spaces)"
echo "======================================================================"

# -----------------------------------------------------------------------------
# Step 1/2: Train  (skip by exporting SKIP_TRAIN=1 to resume at eval only)
# -----------------------------------------------------------------------------
run_train() {
  echo; echo "===== Step 1/2: Train CVLM -> ${OUTPUT_DIR} ====="
  local -a TRAIN_ARGS=(
    "${ROOT}/src/train_cvlm.py"
    --output_dir "${OUTPUT_DIR}"
    --dataset_name "${DATASET_NAME}"
    --model_name_or_path "${MODEL_NAME}"
    --text_encoder_name "${TEXT_ENCODER_NAME}"
    --vision_encoder_name "${VISION_ENCODER_NAME}"
    --compression_rate "${COMPRESSION_RATE}"
    --max_samples "${MAX_SAMPLES}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --max_prompt_len "${MAX_PROMPT_LEN}"
    --max_answer_len "${MAX_ANSWER_LEN}"
    --max_vision_len "${MAX_VISION_LEN}"
    --max_source_len "${MAX_SOURCE_LEN}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --log_interval "${LOG_INTERVAL}"
    --save_interval_steps "${SAVE_INTERVAL_STEPS}"
    --tensorboard_dir "${TB_DIR}/train"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --unfreeze_encoder_top_k "${UNFREEZE_ENCODER_TOP_K}"
    --num_pool_latents "${NUM_POOL_LATENTS}"
    --vision_backend "${VISION_BACKEND}"
    --cr_schedule "${CR_SCHEDULE}"
    --disable_lr_schedule "${DISABLE_LR_SCHEDULE}"
  )
  if [[ "${LEGACY_PROJECTOR}" == "1" ]]; then
    TRAIN_ARGS+=( --legacy_projector true )
  fi
  if [[ "${USE_CHAT_TEMPLATE}" == "1" ]]; then
    TRAIN_ARGS+=( --use_chat_template )
  fi
  if [[ "${ENABLE_WARMUP}" == "1" ]]; then
    TRAIN_ARGS+=( --enable_warmup --warmup_steps "${WARMUP_STEPS}" )
  fi
  if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC}" "${TRAIN_ARGS[@]}"
  else
    python "${TRAIN_ARGS[@]}"
  fi
}

SKIP_TRAIN="${SKIP_TRAIN:-0}"
if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo; echo "===== Step 1/2: SKIPPED (eval-only) ====="
elif compgen -G "${OUTPUT_DIR}/model_step_"*.safetensors >/dev/null 2>&1; then
  echo "ERROR: checkpoints already exist in OUTPUT_DIR=${OUTPUT_DIR}" >&2
  echo "  Refusing to train into an existing run. Use EVAL_ONLY=1 or SKIP_TRAIN=1 for eval," >&2
  echo "  or set OUTPUT_DIR to a new directory to train from scratch." >&2
  exit 1
else
  run_train
fi

# -----------------------------------------------------------------------------
# Step 2/2: Locate latest checkpoint and evaluate
# -----------------------------------------------------------------------------
CKPT=$(ls -1 "${OUTPUT_DIR}"/model_step_*.safetensors 2>/dev/null \
       | sed -E 's|.*model_step_([0-9]+)\.safetensors$|\1 &|' \
       | sort -n \
       | tail -n1 \
       | awk '{print $2}')
if [[ -z "${CKPT}" ]]; then
  echo "ERROR: no model_step_*.safetensors found in ${OUTPUT_DIR}"; exit 1
fi
STEP=$(basename "${CKPT}" | sed -E 's|model_step_([0-9]+)\.safetensors|\1|')
echo; echo "===== Step 2/2: Eval checkpoint ${CKPT} (step=${STEP}) ====="

for MODE in ${EVAL_MODES}; do
  echo; echo "----- eval mode=${MODE} -----"
  EVAL_CMD=(
    python "${ROOT}/src/eval_cvlm.py"
    --checkpoint_path "${CKPT}"
    --dataset_name "${DATASET_NAME}"
    --dataset_split "${EVAL_SPLIT}"
    --model_name_or_path "${MODEL_NAME}"
    --text_encoder_name "${TEXT_ENCODER_NAME}"
    --vision_encoder_name "${VISION_ENCODER_NAME}"
    --compression_rate "${COMPRESSION_RATE}"
    --num_pool_latents "${NUM_POOL_LATENTS}"
    --vision_backend "${VISION_BACKEND}"
    --mode "${MODE}"
    --max_samples "${EVAL_MAX_SAMPLES}"
    --batch_size "${EVAL_BATCH_SIZE}"
    --max_prompt_len "${MAX_PROMPT_LEN}"
    --max_answer_len "${MAX_ANSWER_LEN}"
    --max_vision_len "${MAX_VISION_LEN}"
    --max_source_len "${MAX_SOURCE_LEN}"
    --tensorboard_dir "${TB_DIR}"
    --tb_run_name "eval_${MODE}"
    --global_step "${STEP}"
    --output_json "${OUTPUT_DIR}/eval_${MODE}.json"
    --num_cached_samples "${NUM_CACHED_SAMPLES}"
  )
  if [[ "${LEGACY_PROJECTOR}" == "1" ]]; then
    EVAL_CMD+=( --legacy_projector true )
  fi
  if [[ "${USE_CHAT_TEMPLATE}" == "1" ]]; then
    EVAL_CMD+=( --use_chat_template )
  fi
  if [[ "${EVAL_COMPUTE_GEN}" == "1" ]]; then
    EVAL_CMD+=( --compute_generation_metrics )
  fi
  if [[ "${EVAL_ALL_CHECKPOINTS}" == "1" ]]; then
    EVAL_CMD+=( --all_checkpoints )
  fi
  # Per-ckpt cr lookup is only meaningful for cr-dependent modes; the cvlm/
  # cvlm_shuffle/baseline_proj modes use it. baseline_llm{,_full} ignore cr.
  if [[ -n "${EVAL_CR_SCHEDULE}" && "${EVAL_ALL_CHECKPOINTS}" == "1" ]]; then
    case "${MODE}" in
      cvlm|cvlm_shuffle|baseline_proj)
        EVAL_CMD+=( --cr_schedule "${EVAL_CR_SCHEDULE}" )
        ;;
    esac
  fi
  "${EVAL_CMD[@]}"
done

# ---------------------------------------------------------------------------
# LongBench QA eval (official leaderboard qa_f1_score).
# ---------------------------------------------------------------------------
if [[ "${LONGBENCH_EVAL}" == "1" ]]; then
  echo; echo "===== LongBench QA eval (dataset=${LONGBENCH_DATASET}) ====="
  for MODE in ${EVAL_MODES}; do
    for LB_TASK in ${LONGBENCH_TASKS}; do
      echo; echo "----- LongBench mode=${MODE} task=${LB_TASK} -----"
      LB_CMD=(
        python "${ROOT}/src/eval_cvlm.py"
        --checkpoint_path "${CKPT}"
        --dataset_name "${LONGBENCH_DATASET}"
        --dataset_split "${LONGBENCH_SPLIT}"
        --longbench_task "${LB_TASK}"
        --model_name_or_path "${MODEL_NAME}"
        --text_encoder_name "${TEXT_ENCODER_NAME}"
        --vision_encoder_name "${VISION_ENCODER_NAME}"
        --compression_rate "${COMPRESSION_RATE}"
        --num_pool_latents "${NUM_POOL_LATENTS}"
        --vision_backend "${VISION_BACKEND}"
        --mode "${MODE}"
        --max_samples "${EVAL_MAX_SAMPLES}"
        --batch_size "${EVAL_BATCH_SIZE}"
        --max_prompt_len "${LB_MAX_PROMPT_LEN}"
        --max_answer_len "${MAX_ANSWER_LEN}"
        --max_vision_len "${MAX_VISION_LEN}"
        --max_source_len "${LB_MAX_SOURCE_LEN}"
        --tensorboard_dir "${TB_DIR}"
        --tb_run_name "lb_${MODE}_${LB_TASK}"
        --global_step "${STEP}"
        --output_json "${OUTPUT_DIR}/longbench_${MODE}_${LB_TASK}.json"
        --num_cached_samples "${NUM_CACHED_SAMPLES}"
        --compute_generation_metrics
      )
      if [[ "${LEGACY_PROJECTOR}" == "1" ]]; then
        LB_CMD+=( --legacy_projector true )
      fi
      if [[ "${USE_CHAT_TEMPLATE}" == "1" ]]; then
        LB_CMD+=( --use_chat_template )
      fi
      if [[ "${EVAL_ALL_CHECKPOINTS}" == "1" ]]; then
        LB_CMD+=( --all_checkpoints )
      fi
      # cr-dependent modes get the per-ckpt cr schedule; baseline_llm{,_full} ignore it.
      if [[ -n "${EVAL_CR_SCHEDULE}" && "${EVAL_ALL_CHECKPOINTS}" == "1" ]]; then
        case "${MODE}" in
          cvlm|cvlm_shuffle|baseline_proj)
            LB_CMD+=( --cr_schedule "${EVAL_CR_SCHEDULE}" )
            ;;
        esac
      fi
      "${LB_CMD[@]}"
    done
  done
fi

echo
echo "======================================================================"
echo "DONE. Artifacts:"
echo "  checkpoints : ${OUTPUT_DIR}/model_step_*.safetensors"
echo "  eval JSON   : ${OUTPUT_DIR}/eval_*.json"
echo "  full log    : ${LOG_FILE}"
echo "  trackio UI  : trackio show --project ${TRACKIO_PROJECT}"
echo "======================================================================"