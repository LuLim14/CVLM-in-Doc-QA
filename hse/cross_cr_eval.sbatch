#!/bin/bash
#SBATCH --job-name=cvlm_crosscr
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Cross-CR матрица: каждый чекпоинт ветки × все cr (включая невиданный 32).
# Teacher-forcing метрики + пер-сэмпловый дамп (для bootstrap-CI); генерация
# выключена (дорого для всей матрицы — гоняется отдельно на native-клетках).
# Запуск: sbatch --job-name=crosscr_vit \
#   --export=ALL,RUN_DIR=/home/aburlova/cvlm/runs/abl_vit_<stamp>,BRANCH=vit cross_cr_eval.sbatch
# Опции: CRS="1 2 4 8 16 32" (дефолт), STEPS="1500 3000 ..." (дефолт: все чекпоинты в RUN_DIR).

set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs

module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm
source /home/aburlova/cvlm/hse/protocol.env

RUN_DIR="${RUN_DIR:?set RUN_DIR=/abs/path/to/run}"
BRANCH="${BRANCH:?set BRANCH=vit|randomvit|novit|frozenvit}"
CRS="${CRS:-1 2 4 8 16 32}"

VIT_ARGS=(--vit_init pretrained --freeze_vit 0 --no_vit 0)
case "${BRANCH}" in
  vit)       ;;
  randomvit) VIT_ARGS=(--vit_init random --freeze_vit 0 --no_vit 0) ;;
  novit)     VIT_ARGS=(--vit_init pretrained --freeze_vit 0 --no_vit 1) ;;
  frozenvit) VIT_ARGS=(--vit_init pretrained --freeze_vit 1 --no_vit 0) ;;
  *) echo "unknown BRANCH=${BRANCH}"; exit 1 ;;
esac

if [ -z "${STEPS:-}" ]; then
  STEPS=$(ls -1 "${RUN_DIR}"/model_step_*.safetensors | sed -E 's|.*model_step_([0-9]+)\.safetensors|\1|' | sort -n | tr '\n' ' ')
fi
echo "=== RUN_DIR=${RUN_DIR} BRANCH=${BRANCH} STEPS=${STEPS} CRS=${CRS}"

cd "${CVLM_ROOT}"
export PYTHONPATH="${CVLM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
OUT="${RUN_DIR}/crosscr"
mkdir -p "${OUT}"

for S in ${STEPS}; do
  for CR in ${CRS}; do
    CELL="${OUT}/crosscr_step${S}_cr${CR}"
    if [ -s "${CELL}.json" ]; then echo "skip existing ${CELL}.json"; continue; fi
    echo; echo "----- step=${S} cr=${CR} -----"
    python src/eval_cvlm.py \
      --mode cvlm \
      --checkpoint_path "${RUN_DIR}/model_step_${S}.safetensors" \
      --dataset_name sggetao/PwC \
      --dataset_split "${EVAL_SPLIT}" \
      --model_name_or_path "${MODEL_NAME}" \
      --text_encoder_name "${TEXT_ENCODER_NAME}" \
      --vision_encoder_name "${VISION_ENCODER_NAME}" \
      --compression_rate "${CR}" \
      --num_pool_latents "${NUM_POOL_LATENTS}" \
      --max_prompt_len "${MAX_PROMPT_LEN}" \
      --max_answer_len "${MAX_ANSWER_LEN}" \
      --max_vision_len "${MAX_VISION_LEN}" \
      --max_source_len "${MAX_SOURCE_LEN}" \
      --max_samples "${EVAL_MAX_SAMPLES}" \
      --batch_size "${EVAL_BATCH_SIZE}" \
      --global_step "${S}" \
      --seed "${SEED}" \
      --output_json "${CELL}.json" \
      --dump_per_sample "${CELL}.samples.jsonl" \
      "${VIT_ARGS[@]}"
  done
done
echo "=== CROSS-CR DONE: ${OUT}"
