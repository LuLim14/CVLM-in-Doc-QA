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
# Mechanistic audit artifacts: a fixed tokenized probe and the exact step-0
# compression-path weights. These make CKA/weight-drift analysis possible from
# saved checkpoints without recreating the dataset or random initialization.
MECHANISTIC_PROBE_SAMPLES="${MECHANISTIC_PROBE_SAMPLES:-16}"
SAVE_MECHANISTIC_INIT="${SAVE_MECHANISTIC_INIT:-1}"
NPROC="${NPROC:-1}"
# Linear warmup stabilises the first ~100 optimizer steps. With the projectors
# + ViT freshly initialised, early grad norms can spike above 100; without
# warmup clip=1.0 silently discards most of those updates.
ENABLE_WARMUP="${ENABLE_WARMUP:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
UNFREEZE_ENCODER_TOP_K="${UNFREEZE_ENCODER_TOP_K:-22}"   # 0 = encoder fully frozen; >0 unfreezes top-K layers + final_norm, 22 == fully unreeze for ModerBert
NUM_POOL_LATENTS="${NUM_POOL_LATENTS:-4}"                # K learnable queries per chunk for the attention pool; K>1 = Perceiver-style multi-query pool whose K outputs are mean-pooled to 1 vision token per chunk
# MEANPOOL ABLATION (MeanPool+Project): 1 = the chunk pool is an EXACT masked
# mean with no learnable part (pool_queries stays allocated but frozen at zero).
# Passed to BOTH train and eval — a mismatch silently evaluates a different pool.
MEAN_POOL="${MEAN_POOL:-0}"

# 1 = plain projectors (no input_norm.* in safetensors — old runs before RMSNorm)
LEGACY_PROJECTOR="${LEGACY_PROJECTOR:-0}"
export LEGACY_PROJECTOR

CR_SCHEDULE="${CR_SCHEDULE:-1:6000,2:12000,4:18000,8:24000,16:0}"   # empty = static cr from COMPRESSION_RATE
DISABLE_LR_SCHEDULE="${DISABLE_LR_SCHEDULE:-1}"   # 1 = constant LR after warmup (no cosine decay)

# NUMERICAL-PRECISION control. 1 = AdamW keeps fp32 master copies of the bf16
# trainable weights (params stay bf16 for forward/backward; updates smaller than
# half a bf16 ULP accumulate in the master instead of being rounded away). With
# 0 (legacy) an update of size ~lr=1e-4 cannot move a ~1.0-magnitude pretrained
# ViT weight at all, which confounds the pretrained-vs-random comparison.
FP32_MASTER_WEIGHTS="${FP32_MASTER_WEIGHTS:-0}"
# Scope of that treatment: all = every low-precision trainable tensor (~1.84B
# params here — the decoder IS trainable — i.e. ~20.5 GiB of extra optimizer
# state and four sub-modules changing numerics at once); vision = ONLY the
# ViT/mixer stack (~86M, ~1.0 GiB), the isolated control.
FP32_MASTER_SCOPE="${FP32_MASTER_SCOPE:-all}"

# ViT ablation branch (prior study): VIT_INIT=pretrained|random, FREEZE_VIT=0|1, NO_VIT=0|1.
VIT_INIT="${VIT_INIT:-pretrained}"
FREEZE_VIT="${FREEZE_VIT:-0}"
NO_VIT="${NO_VIT:-0}"
# POST-BOTTLENECK DEPTH: 0 = the full 12-layer ViT; N>0 keeps only the first N
# blocks (same block type / init draw / final LayerNorm as the 12-layer arm, so
# a 0/N/12 curve differs in depth ONLY). Passed to BOTH train and eval.
VIT_NUM_LAYERS="${VIT_NUM_LAYERS:-0}"

# RATE-DEPTH MIXER: non-empty "cr:depth,..." map replaces the ViT with a stack
# of zero-gated ViTLayer blocks (depth per compression rate). Passed to BOTH
# train and eval. Incompatible with NO_VIT/FREEZE_VIT/VIT_INIT=random.
MIXER_DEPTH_MAP="${MIXER_DEPTH_MAP:-}"
# Padding-safe ViT/mixer attention. Keep 0 only for exact reproduction of the
# legacy unmasked runs; new controlled experiments should set 1 for every arm.
MASK_VIT_PADDING="${MASK_VIT_PADDING:-0}"
# Plain ViTModel applies a final LayerNorm after encoder.layer; the bare mixer
# stack does not. Set 0 for a direct-vs-zero-gated control with matched blocks.
VIT_APPLY_FINAL_LAYERNORM="${VIT_APPLY_FINAL_LAYERNORM:-1}"
# Compact decoder positions across left-padding and internal masked latent
# slots. Generation already does this; enabling it aligns teacher forcing.
COMPACT_DECODER_POSITIONS="${COMPACT_DECODER_POSITIONS:-0}"
# MIXED-CR training: non-empty whitespace-separated cr set (e.g. "1 2 4 8 16")
# samples one cr per optimizer step from a balanced deterministic sequence.
# Replaces the CR_SCHEDULE curriculum (an empty --cr_schedule is forced below).
MIXED_CR_SET="${MIXED_CR_SET:-}"

# Optional short tag baked into the default run-dir name (e.g. abl_randomvit).
RUN_TAG="${RUN_TAG:-cvlm}"
# Root for run dirs; override RUNS_ROOT or the full OUTPUT_DIR.
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"
# Output dir — default name embeds main hyperparameters for easy A/B comparison.
# Override by exporting OUTPUT_DIR=/abs/path before running the script.
OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${RUN_TAG}_cr${COMPRESSION_RATE}_vl${MAX_VISION_LEN}_sl${MAX_SOURCE_LEN}_bs${BATCH_SIZE}_lr${LR}_ep${EPOCHS}_gc${GRADIENT_CHECKPOINTING}_unfrzed${UNFREEZE_ENCODER_TOP_K}_lt${NUM_POOL_LATENTS}}"

# Resume: export RESUME_DIR=/abs/path/to/run to continue from its latest
# checkpoint pair (needed to survive partition walltime limits).
RESUME_DIR="${RESUME_DIR:-}"

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
if [[ -n "${MIXED_CR_SET}" ]]; then
  # Mixed-cr training has no per-step "which cr trained this checkpoint"
  # mapping, so the native-cr eval sweep would be wrong — force it off.
  EVAL_CR_SCHEDULE=""
fi

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
LOG_FILE="${OUTPUT_DIR}/pipeline.log"

mkdir -p "${OUTPUT_DIR}" "${TB_DIR}"

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
echo "  MECHANISTIC_PROBE_SAMPLES = ${MECHANISTIC_PROBE_SAMPLES}"
echo "  SAVE_MECHANISTIC_INIT = ${SAVE_MECHANISTIC_INIT}"
echo "  ENABLE_WARMUP     = ${ENABLE_WARMUP} (steps=${WARMUP_STEPS})"
echo "  GRADIENT_CHECKPOINTING = ${GRADIENT_CHECKPOINTING}"
echo "  UNFREEZE_ENCODER_TOP_K = ${UNFREEZE_ENCODER_TOP_K}"
echo "  NUM_POOL_LATENTS  = ${NUM_POOL_LATENTS}"
echo "  MEAN_POOL         = ${MEAN_POOL}"
echo "  VIT_INIT          = ${VIT_INIT}"
echo "  VIT_NUM_LAYERS    = ${VIT_NUM_LAYERS} (0 = all)"
echo "  FREEZE_VIT        = ${FREEZE_VIT}"
echo "  NO_VIT            = ${NO_VIT}"
echo "  MIXER_DEPTH_MAP   = ${MIXER_DEPTH_MAP:-<off>}"
echo "  MASK_VIT_PADDING  = ${MASK_VIT_PADDING}"
echo "  VIT_APPLY_FINAL_LAYERNORM = ${VIT_APPLY_FINAL_LAYERNORM}"
echo "  COMPACT_DECODER_POSITIONS = ${COMPACT_DECODER_POSITIONS}"
echo "  MIXED_CR_SET      = ${MIXED_CR_SET:-<off>}"
echo "  FP32_MASTER_WEIGHTS = ${FP32_MASTER_WEIGHTS} (scope=${FP32_MASTER_SCOPE})"
echo "  RESUME_DIR        = ${RESUME_DIR:-<none>}"
echo "  LEGACY_PROJECTOR  = ${LEGACY_PROJECTOR}"
echo "  CR_SCHEDULE       = ${CR_SCHEDULE:-<static>}"
echo "  DISABLE_LR_SCHEDULE = ${DISABLE_LR_SCHEDULE}"
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
# Resolved architecture flags on disk, independent of any sbatch wrapper: these
# are the values THIS invocation actually forwards to train_cvlm/eval_cvlm, so a
# run directory always carries the arm it really ran (env vars read directly by
# python would otherwise leave no artifact).
{
  echo "MEAN_POOL=${MEAN_POOL}"
  echo "VIT_NUM_LAYERS=${VIT_NUM_LAYERS}"
  echo "VIT_INIT=${VIT_INIT}"
  echo "FREEZE_VIT=${FREEZE_VIT}"
  echo "NO_VIT=${NO_VIT}"
  echo "MIXER_DEPTH_MAP=${MIXER_DEPTH_MAP}"
  echo "MASK_VIT_PADDING=${MASK_VIT_PADDING}"
  echo "VIT_APPLY_FINAL_LAYERNORM=${VIT_APPLY_FINAL_LAYERNORM}"
  echo "COMPACT_DECODER_POSITIONS=${COMPACT_DECODER_POSITIONS}"
  echo "NUM_POOL_LATENTS=${NUM_POOL_LATENTS}"
  echo "CR_SCHEDULE=${CR_SCHEDULE}"
  echo "MIXED_CR_SET=${MIXED_CR_SET}"
  echo "COMPRESSION_RATE=${COMPRESSION_RATE}"
  echo "EPOCHS=${EPOCHS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "GRAD_ACCUM=${GRAD_ACCUM}"
  echo "LR=${LR}"
  echo "SEED=${SEED:-42}"
} > "${OUTPUT_DIR}/PIPELINE_FLAGS"

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
    --mechanistic_probe_samples "${MECHANISTIC_PROBE_SAMPLES}"
    --save_mechanistic_init "${SAVE_MECHANISTIC_INIT}"
    --tensorboard_dir "${TB_DIR}/train"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --unfreeze_encoder_top_k "${UNFREEZE_ENCODER_TOP_K}"
    --num_pool_latents "${NUM_POOL_LATENTS}"
    --mean_pool "${MEAN_POOL}"
    --disable_lr_schedule "${DISABLE_LR_SCHEDULE}"
    --vit_init "${VIT_INIT}"
    --vit_num_layers "${VIT_NUM_LAYERS}"
    --freeze_vit "${FREEZE_VIT}"
    --no_vit "${NO_VIT}"
    --mask_vit_padding "${MASK_VIT_PADDING}"
    --vit_apply_final_layernorm "${VIT_APPLY_FINAL_LAYERNORM}"
    --compact_decoder_positions "${COMPACT_DECODER_POSITIONS}"
    --fp32_master_weights "${FP32_MASTER_WEIGHTS}"
    --fp32_master_scope "${FP32_MASTER_SCOPE}"
    --seed "${SEED:-42}"
  )
  if [[ -n "${MIXED_CR_SET}" ]]; then
    # Mixed-cr replaces the stage curriculum. --cr_schedule "" is passed
    # EXPLICITLY (not just omitted) so an exported CR_SCHEDULE env var cannot
    # leak back in through the argparse env-default and trip the
    # mixed_cr_set/cr_schedule incompatibility check.
    TRAIN_ARGS+=( --mixed_cr_set "${MIXED_CR_SET}" --cr_schedule "" )
  else
    TRAIN_ARGS+=( --cr_schedule "${CR_SCHEDULE}" )
  fi
  if [[ -n "${MIXER_DEPTH_MAP}" ]]; then
    TRAIN_ARGS+=( --mixer_depth_map "${MIXER_DEPTH_MAP}" )
  fi
  if [[ "${LEGACY_PROJECTOR}" == "1" ]]; then
    TRAIN_ARGS+=( --legacy_projector true )
  fi
  if [[ "${ENABLE_WARMUP}" == "1" ]]; then
    TRAIN_ARGS+=( --enable_warmup --warmup_steps "${WARMUP_STEPS}" )
  fi
  if [[ -n "${RESUME_DIR}" ]]; then
    TRAIN_ARGS+=( --resume_dir "${RESUME_DIR}" )
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
elif [[ -z "${RESUME_DIR}" ]] && compgen -G "${OUTPUT_DIR}/model_step_"*.safetensors >/dev/null 2>&1; then
  echo "ERROR: checkpoints already exist in OUTPUT_DIR=${OUTPUT_DIR}" >&2
  echo "  Refusing to train into an existing run. Use EVAL_ONLY=1 or SKIP_TRAIN=1 for eval," >&2
  echo "  set RESUME_DIR=${OUTPUT_DIR} to continue it, or point OUTPUT_DIR at a new directory." >&2
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
    --mean_pool "${MEAN_POOL}"
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
    --vit_init "${VIT_INIT}"
    --vit_num_layers "${VIT_NUM_LAYERS}"
    --freeze_vit "${FREEZE_VIT}"
    --no_vit "${NO_VIT}"
    --mask_vit_padding "${MASK_VIT_PADDING}"
    --vit_apply_final_layernorm "${VIT_APPLY_FINAL_LAYERNORM}"
    --compact_decoder_positions "${COMPACT_DECODER_POSITIONS}"
    --seed "${SEED:-42}"
  )
  if [[ -n "${MIXER_DEPTH_MAP}" ]]; then
    EVAL_CMD+=( --mixer_depth_map "${MIXER_DEPTH_MAP}" )
  fi
  if [[ "${EVAL_DUMP:-0}" == "1" ]]; then
    EVAL_CMD+=( --dump_per_sample "${OUTPUT_DIR}/eval_${MODE}_dump.jsonl" )
  fi
  if [[ "${LEGACY_PROJECTOR}" == "1" ]]; then
    EVAL_CMD+=( --legacy_projector true )
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

echo
echo "======================================================================"
echo "DONE. Artifacts:"
echo "  checkpoints : ${OUTPUT_DIR}/model_step_*.safetensors"
echo "  eval JSON   : ${OUTPUT_DIR}/eval_*.json"
echo "  full log    : ${LOG_FILE}"
echo "  trackio UI  : trackio show --project ${TRACKIO_PROJECT}"
echo "======================================================================"
