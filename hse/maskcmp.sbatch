#!/bin/bash
#SBATCH --job-name=cvlm_mask
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# РЕШАЮЩИЙ ЭКСПЕРИМЕНТ ПРО КОНФАУНДЕР: снимает ли pad-маскирование штраф за глубину?
#   ARM=novit    : pooling-only (эталон; ViT-путь не задействован)
#   ARM=unmasked : random-init 12-слойный ViT БЕЗ маски (воспроизводит боевую ветку)
#   ARM=masked   : тот же ViT С key-padding маской
# Единственная разница unmasked/masked — маска. Всё остальное идентично
# (seed, батчи, shared_init_seed, куррикулум).
set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs
module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm
SNAPSHOT_ROOT="${CVLM_SNAPSHOT_ROOT:-/home/aburlova/cvlm_rd5}"
source "${SNAPSHOT_ROOT}/hse/protocol.env"
export CVLM_ROOT="${SNAPSHOT_ROOT}"
ARM="${ARM:?set ARM=novit|unmasked|masked}"
export VIT_INIT=random FREEZE_VIT=0 NO_VIT=0 MASK_VIT_PADDING=0 MIXER_DEPTH_MAP=""
# Match the direct stack to the no-ViT path: the bare zero-gated mixer has no
# ViTModel final LayerNorm, so keeping it here would be a second architectural
# difference besides the 12 post-bottleneck blocks.
export VIT_APPLY_FINAL_LAYERNORM=0
export COMPACT_DECODER_POSITIONS=0
case "$ARM" in
  novit)    export NO_VIT=1 ;;
  unmasked) export MASK_VIT_PADDING=0 ;;
  masked)   export MASK_VIT_PADDING=1 ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac
export RUN_TAG="mask_${ARM}"
export OUTPUT_DIR="${RUNS_ROOT}/mask_${ARM}_${MC_STAMP:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"
cp "${CVLM_ROOT}/VERSION" "${OUTPUT_DIR}/CODE_VERSION"
cd "${CVLM_ROOT}"
sha256sum \
  src/modeling.py src/train_cvlm.py src/train_utils.py src/train_logging.py \
  src/eval_cvlm.py scripts/run_full_pipeline.sh hse/maskcmp.sbatch hse/protocol.env \
  > "${OUTPUT_DIR}/CODE_SHA256"

# The generic pipeline performs one cheap cr=1 sanity eval. The actual result
# is the paired cross-CR sweep below; do not accidentally spend the walltime on
# every checkpoint or generation for all six rates.
MASKCMP_EVAL_MAX_SAMPLES="${MASKCMP_EVAL_MAX_SAMPLES:-${EVAL_MAX_SAMPLES}}"
MASKCMP_GEN_CRS="${MASKCMP_GEN_CRS:-16}"
export EVAL_ALL_CHECKPOINTS=0
export EVAL_MODES=cvlm
export EVAL_COMPUTE_GEN=0
export EVAL_MAX_SAMPLES="${MASKCMP_SANITY_SAMPLES:-64}"

{
  echo "ARM=${ARM}"
  echo "MASK_VIT_PADDING=${MASK_VIT_PADDING}"
  echo "NO_VIT=${NO_VIT}"
  echo "VIT_INIT=${VIT_INIT}"
  echo "VIT_APPLY_FINAL_LAYERNORM=${VIT_APPLY_FINAL_LAYERNORM}"
  echo "COMPACT_DECODER_POSITIONS=${COMPACT_DECODER_POSITIONS}"
  echo "MASKCMP_EVAL_MAX_SAMPLES=${MASKCMP_EVAL_MAX_SAMPLES}"
  echo "MASKCMP_GEN_CRS=${MASKCMP_GEN_CRS}"
  echo "SEED=${SEED}"
} > "${OUTPUT_DIR}/EXPERIMENT_FLAGS"

echo "=== ARM=${ARM} MASK_VIT_PADDING=${MASK_VIT_PADDING} NO_VIT=${NO_VIT} OUT=${OUTPUT_DIR}"
bash scripts/run_full_pipeline.sh

LAST=$(
  find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'model_step_*.safetensors' \
    | sed -E 's|.*model_step_([0-9]+)\.safetensors|\1 &|' \
    | sort -n \
    | tail -1 \
    | awk '{print $2}'
)
if [[ -z "${LAST}" ]]; then
  echo "ERROR: no final checkpoint in ${OUTPUT_DIR}" >&2
  exit 1
fi
STEP=$(basename "${LAST}" | grep -oE '[0-9]+')

for CR in 1 2 4 8 16 32; do
  EVAL_ARGS=(
    src/eval_cvlm.py --mode cvlm
    --checkpoint_path "${LAST}" --compression_rate "${CR}"
    --dataset_name sggetao/PwC --dataset_split "${EVAL_SPLIT}"
    --model_name_or_path "${MODEL_NAME}" --text_encoder_name "${TEXT_ENCODER_NAME}"
    --vision_encoder_name "${VISION_ENCODER_NAME}"
    --num_pool_latents "${NUM_POOL_LATENTS}" --max_prompt_len "${MAX_PROMPT_LEN}"
    --max_answer_len "${MAX_ANSWER_LEN}" --max_vision_len "${MAX_VISION_LEN}"
    --max_source_len "${MAX_SOURCE_LEN}" --max_samples "${MASKCMP_EVAL_MAX_SAMPLES}"
    --batch_size "${EVAL_BATCH_SIZE}" --seed "${SEED}" --global_step "${STEP}"
    --vit_init "${VIT_INIT}" --freeze_vit "${FREEZE_VIT}" --no_vit "${NO_VIT}"
    --mask_vit_padding "${MASK_VIT_PADDING}"
    --vit_apply_final_layernorm "${VIT_APPLY_FINAL_LAYERNORM}"
    --compact_decoder_positions "${COMPACT_DECODER_POSITIONS}"
    --output_json "${OUTPUT_DIR}/eval_maskcmp_cr${CR}.json"
    --dump_per_sample "${OUTPUT_DIR}/eval_maskcmp_cr${CR}_dump.jsonl"
  )
  if [[ " ${MASKCMP_GEN_CRS} " == *" ${CR} "* ]]; then
    EVAL_ARGS+=( --compute_generation_metrics )
  fi
  python "${EVAL_ARGS[@]}"
done

sha256sum -c "${OUTPUT_DIR}/CODE_SHA256"
ls -1 "${OUTPUT_DIR}"/trainer_step_*.pt 2>/dev/null \
  | sort -t_ -k3 -n \
  | head -n -1 \
  | xargs -r rm -f
echo "=== MASKCMP DONE ${ARM}"
