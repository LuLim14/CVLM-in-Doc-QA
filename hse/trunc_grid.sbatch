#!/bin/bash
#SBATCH --job-name=cvlm_trunc
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Inference-time truncation-сетка (RQ3): frozen декодер получает вопрос + окно
# документа при бюджете N = числу латентов CVLM на данном cr.
# Стратегии: first/last/random/oracle (окно вокруг найденного ответа, hit-rate в JSON).
# Опции --export: MODE=baseline_llm_full (дефолт, frozen SmolLM2) или MODE=sft
#   c SFT_PATH=/abs/path (truncated-SFT декодер, когда доедет sft_trunc*);
#   BUDGET_CRS="4 8 16" (дефолт), STRATEGIES="first last random oracle" (дефолт),
#   GEN=1 (генеративные метрики; дефолт 1 — на капчёном EVAL_MAX_SAMPLES это ок).

set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs

module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm
source /home/aburlova/cvlm/hse/protocol.env

MODE="${MODE:-baseline_llm_full}"
SFT_PATH="${SFT_PATH:-}"
BUDGET_CRS="${BUDGET_CRS:-4 8 16}"
STRATEGIES="${STRATEGIES:-first last random oracle}"
GEN="${GEN:-1}"

OUT="${RUNS_ROOT}/trunc_grid_${MODE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT}"
cd "${CVLM_ROOT}"
export PYTHONPATH="${CVLM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

EXTRA=()
[ "${MODE}" = "sft" ] && EXTRA+=( --sft_model_path "${SFT_PATH:?MODE=sft requires SFT_PATH}" )
[ "${GEN}" = "1" ] && EXTRA+=( --compute_generation_metrics )

for CR in ${BUDGET_CRS}; do
  for STRAT in ${STRATEGIES}; do
    CELL="${OUT}/trunc_${MODE}_${STRAT}_cr${CR}"
    if [ -s "${CELL}.json" ]; then echo "skip ${CELL}"; continue; fi
    echo; echo "----- mode=${MODE} strat=${STRAT} budget_cr=${CR} -----"
    python src/eval_cvlm.py \
      --mode "${MODE}" \
      --dataset_name sggetao/PwC \
      --dataset_split "${EVAL_SPLIT}" \
      --model_name_or_path "${MODEL_NAME}" \
      --text_encoder_name "${TEXT_ENCODER_NAME}" \
      --vision_encoder_name "${VISION_ENCODER_NAME}" \
      --compression_rate 1 \
      --num_pool_latents "${NUM_POOL_LATENTS}" \
      --max_prompt_len "${MAX_PROMPT_LEN}" \
      --max_answer_len "${MAX_ANSWER_LEN}" \
      --max_vision_len "${MAX_VISION_LEN}" \
      --max_source_len "${MAX_SOURCE_LEN}" \
      --max_samples "${EVAL_MAX_SAMPLES}" \
      --batch_size "${EVAL_BATCH_SIZE}" \
      --seed "${SEED}" \
      --doc_truncation "${STRAT}" \
      --doc_budget_cr "${CR}" \
      --output_json "${CELL}.json" \
      --dump_per_sample "${CELL}.samples.jsonl" \
      "${EXTRA[@]}"
  done
done
# Референс без усечения (полный документ) для тех же сэмплов:
REF="${OUT}/trunc_${MODE}_none_full"
if [ ! -s "${REF}.json" ]; then
  python src/eval_cvlm.py \
    --mode "${MODE}" \
    --dataset_name sggetao/PwC --dataset_split "${EVAL_SPLIT}" \
    --model_name_or_path "${MODEL_NAME}" --text_encoder_name "${TEXT_ENCODER_NAME}" \
    --vision_encoder_name "${VISION_ENCODER_NAME}" --compression_rate 1 \
    --num_pool_latents "${NUM_POOL_LATENTS}" --max_prompt_len "${MAX_PROMPT_LEN}" \
    --max_answer_len "${MAX_ANSWER_LEN}" --max_vision_len "${MAX_VISION_LEN}" \
    --max_source_len "${MAX_SOURCE_LEN}" --max_samples "${EVAL_MAX_SAMPLES}" \
    --batch_size "${EVAL_BATCH_SIZE}" --seed "${SEED}" \
    --output_json "${REF}.json" --dump_per_sample "${REF}.samples.jsonl" \
    "${EXTRA[@]}"
fi
echo "=== TRUNC GRID DONE: ${OUT}"
