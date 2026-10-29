#!/bin/bash
#SBATCH --job-name=cvlm_paddiag
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Causal diagnostic for the original depth result, using existing checkpoints:
#   1. batch-size and batch-composition sensitivity;
#   2. synthetic padding appended after pooling;
#   3. attention mass assigned to padded keys;
#   4. legacy vs surgical mask-safe inference with identical learned weights.
#
# No retraining is performed. Each JSON is idempotent and carries the requested
# mask mode; source hashes and the exact command log live beside the results.
set -euo pipefail

mkdir -p /home/aburlova/cvlm/runs/_logs
module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm

SNAPSHOT_ROOT="${CVLM_SNAPSHOT_ROOT:-/home/aburlova/cvlm_rd5}"
source "${SNAPSHOT_ROOT}/hse/protocol.env"
export CVLM_ROOT="${SNAPSHOT_ROOT}"
DATASET_NAME="${DATASET_NAME:-sggetao/PwC}"
cd "${CVLM_ROOT}"
export PYTHONPATH="${CVLM_ROOT}/src"
export PYTHONUNBUFFERED=1

CODE_VERSION=$(cat "${CVLM_ROOT}/VERSION")
OUT="${PAD_DIAG_OUT:-/home/aburlova/cvlm/runs/padding_diag_${CODE_VERSION}}"
mkdir -p "${OUT}"
cp "${CVLM_ROOT}/VERSION" "${OUT}/CODE_VERSION"
sha256sum \
  src/modeling.py tools/measure_padding_contamination.py hse/padding_diag.sbatch \
  > "${OUT}/CODE_SHA256"

# Use one COMMON checkpoint step across all three branches. Selecting each
# branch's latest checkpoint independently would confound architecture with
# training progress when the live jobs finish at different times.
RUN_NOVIT=/home/aburlova/cvlm/runs/abl_novit_20260729_171933
RUN_RANDOMVIT=/home/aburlova/cvlm/runs/abl_randomvit_20260729_171933
RUN_VIT=/home/aburlova/cvlm/runs/abl_vit_20260729_171933
COMMON_STEP="${PAD_DIAG_STEP:-}"
if [[ -z "${COMMON_STEP}" ]]; then
  while IFS= read -r CANDIDATE; do
    STEP=$(basename "${CANDIDATE}" | grep -oE '[0-9]+')
    if (
      [[ -s "${RUN_RANDOMVIT}/model_step_${STEP}.safetensors" ]] \
        && [[ -s "${RUN_VIT}/model_step_${STEP}.safetensors" ]]
    ); then
      COMMON_STEP="${STEP}"
      break
    fi
  done < <(
    find "${RUN_NOVIT}" -maxdepth 1 -type f -name 'model_step_*.safetensors' \
      | sed -E 's|.*model_step_([0-9]+)\.safetensors|\1 &|' \
      | sort -nr \
      | awk '{print $2}'
  )
fi
if [[ -z "${COMMON_STEP}" ]]; then
  echo "ERROR: branches have no common checkpoint step" >&2
  exit 1
fi

{
  echo "COMMON_STEP=${COMMON_STEP}"
  echo "MAX_SAMPLES=${PAD_DIAG_MAX_SAMPLES:-128}"
  echo "BATCH1_SAMPLES=${PAD_DIAG_BATCH1_SAMPLES:-32}"
  echo "ATTENTION_BATCHES=${PAD_DIAG_ATTENTION_BATCHES:-2}"
  echo "ATTENTION_QUERY_CHUNK=${PAD_DIAG_ATTENTION_QUERY_CHUNK:-128}"
  echo "PAD_FRACTIONS=${PAD_DIAG_PAD_FRACTIONS:-0,0.25,0.5,0.75}"
  echo "SEED=${SEED}"
} > "${OUT}/DIAGNOSTIC_FLAGS"

for BRANCH in novit randomvit vit; do
  RUN_DIR="/home/aburlova/cvlm/runs/abl_${BRANCH}_20260729_171933"
  CKPT="${RUN_DIR}/model_step_${COMMON_STEP}.safetensors"
  if [[ ! -s "${CKPT}" ]]; then
    echo "ERROR: missing paired checkpoint ${CKPT}" >&2
    exit 1
  fi

  VIT_INIT=pretrained
  NO_VIT=false
  case "${BRANCH}" in
    randomvit) VIT_INIT=random ;;
    novit) NO_VIT=true ;;
  esac

  MASK_MODES=(false true)
  if [[ "${BRANCH}" == "novit" ]]; then
    # The flag is a no-op without a post-bottleneck stack.
    MASK_MODES=(false)
  fi

  for CR in 1 16; do
    for MASK_MODE in "${MASK_MODES[@]}"; do
      RESULT="${OUT}/${BRANCH}_cr${CR}_mask${MASK_MODE}.json"
      if [[ -s "${RESULT}" ]]; then
        echo "SKIP existing ${RESULT}"
        continue
      fi
      echo "=== branch=${BRANCH} cr=${CR} mask_vit_padding=${MASK_MODE}"
      python tools/measure_padding_contamination.py \
        --checkpoint_path "${CKPT}" \
        --compression_rate "${CR}" \
        --model_name_or_path "${MODEL_NAME}" \
        --text_encoder_name "${TEXT_ENCODER_NAME}" \
        --vision_encoder_name "${VISION_ENCODER_NAME}" \
        --dataset_name "${DATASET_NAME}" \
        --dataset_split "${EVAL_SPLIT}" \
        --num_pool_latents "${NUM_POOL_LATENTS}" \
        --max_vision_len "${MAX_VISION_LEN}" \
        --max_source_len "${MAX_SOURCE_LEN}" \
        --max_prompt_len "${MAX_PROMPT_LEN}" \
        --max_answer_len "${MAX_ANSWER_LEN}" \
        --max_samples "${PAD_DIAG_MAX_SAMPLES:-128}" \
        --batch1_samples "${PAD_DIAG_BATCH1_SAMPLES:-32}" \
        --vit_init "${VIT_INIT}" \
        --no_vit "${NO_VIT}" \
        --mask_vit_padding "${MASK_MODE}" \
        --vit_apply_final_layernorm true \
        --compact_decoder_positions false \
        --attention_batches "${PAD_DIAG_ATTENTION_BATCHES:-2}" \
        --attention_query_chunk "${PAD_DIAG_ATTENTION_QUERY_CHUNK:-128}" \
        --pad_fractions "${PAD_DIAG_PAD_FRACTIONS:-0,0.25,0.5,0.75}" \
        --seed "${SEED}" \
        --output_json "${RESULT}"
    done
  done
done

sha256sum -c "${OUT}/CODE_SHA256"
echo "PADDING_DIAG_DONE"
