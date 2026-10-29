#!/bin/bash
#SBATCH --job-name=cvlm_posdiag
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Same no-ViT checkpoint, legacy absolute positions vs mask-compacted decoder
# positions. This isolates the positional effect of masked latent holes without
# any ViT attention, retraining, or change to learned weights.
set -euo pipefail

mkdir -p /home/aburlova/cvlm/runs/_logs
module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm

SNAPSHOT_ROOT="${CVLM_SNAPSHOT_ROOT:-/home/aburlova/cvlm_rd6}"
source "${SNAPSHOT_ROOT}/hse/protocol.env"
export CVLM_ROOT="${SNAPSHOT_ROOT}"
DATASET_NAME="${DATASET_NAME:-sggetao/PwC}"
cd "${CVLM_ROOT}"
export PYTHONPATH="${CVLM_ROOT}/src"
export PYTHONUNBUFFERED=1

RUN_DIR=/home/aburlova/cvlm/runs/abl_novit_20260729_171933
STEP="${POSITION_DIAG_STEP:-}"
if [[ -z "${STEP}" ]]; then
  CKPT=$(
    find "${RUN_DIR}" -maxdepth 1 -type f -name 'model_step_*.safetensors' \
      | sed -E 's|.*model_step_([0-9]+)\.safetensors|\1 &|' \
      | sort -n \
      | tail -1 \
      | awk '{print $2}'
  )
  STEP=$(basename "${CKPT}" | grep -oE '[0-9]+')
else
  CKPT="${RUN_DIR}/model_step_${STEP}.safetensors"
fi
if [[ ! -s "${CKPT}" ]]; then
  echo "ERROR: missing checkpoint ${CKPT}" >&2
  exit 1
fi

CODE_VERSION=$(cat VERSION)
OUT="${POSITION_DIAG_OUT:-/home/aburlova/cvlm/runs/position_diag_${CODE_VERSION}_step${STEP}}"
mkdir -p "${OUT}"
cp VERSION "${OUT}/CODE_VERSION"
sha256sum \
  src/modeling.py src/train_utils.py tools/measure_padding_contamination.py \
  hse/position_diag.sbatch \
  > "${OUT}/CODE_SHA256"
{
  echo "CHECKPOINT=${CKPT}"
  echo "STEP=${STEP}"
  echo "MAX_SAMPLES=${POSITION_DIAG_MAX_SAMPLES:-128}"
  echo "BATCH1_SAMPLES=${POSITION_DIAG_BATCH1_SAMPLES:-32}"
  echo "PAD_FRACTIONS=${POSITION_DIAG_PAD_FRACTIONS:-0,0.25,0.5,0.75}"
  echo "SEED=${SEED}"
} > "${OUT}/DIAGNOSTIC_FLAGS"

for CR in 1 16; do
  for COMPACT in false true; do
    RESULT="${OUT}/novit_cr${CR}_compact${COMPACT}.json"
    if [[ -s "${RESULT}" ]]; then
      echo "SKIP existing ${RESULT}"
      continue
    fi
    echo "=== no-ViT cr=${CR} compact_decoder_positions=${COMPACT}"
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
      --max_samples "${POSITION_DIAG_MAX_SAMPLES:-128}" \
      --batch1_samples "${POSITION_DIAG_BATCH1_SAMPLES:-32}" \
      --vit_init pretrained \
      --no_vit true \
      --mask_vit_padding false \
      --vit_apply_final_layernorm true \
      --compact_decoder_positions "${COMPACT}" \
      --attention_batches 0 \
      --pad_fractions "${POSITION_DIAG_PAD_FRACTIONS:-0,0.25,0.5,0.75}" \
      --seed "${SEED}" \
      --output_json "${RESULT}"
  done
done

sha256sum -c "${OUT}/CODE_SHA256"
echo "POSITION_DIAG_DONE"
