#!/bin/bash
#SBATCH --job-name=cvlm_rd
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Rate-Conditional Depth: один арм эксперимента «Harder Compression, Deeper
# Compressor». Все армы обучаются в MIXED-CR режиме (cr сэмплируется на батч
# из MIXED_CR_SET) с битово одинаковыми общими модулями (shared_init_seed).
# Арм задаётся картой глубины пост-bottleneck миксера (zero-gated ViT-слои):
#   ARM=novit          : NO_VIT=1 (pooling-only, без миксера)
#   ARM=fixed12direct  : VIT_INIT=random (12 слоёв БЕЗ гейтов — текущая архитектура)
#   ARM=fixed12anchor  : MIXER_DEPTH_MAP="1:12,2:12,4:12,8:12,16:12" (zero-gated residual)
#   ARM=ratedepth      : MIXER_DEPTH_MAP="1:0,2:1,4:3,8:6,16:12"    (условная глубина)
# Пары сравнения: direct-vs-anchor изолирует эффект function-preserving insertion;
# anchor-vs-ratedepth изолирует эффект условной глубины.
# Запуск: sbatch --job-name=rd_ratedepth --export=ALL,ARM=ratedepth ratedepth.sbatch
# Интерфейс кода (реализуется в modeling.py/train_cvlm.py/eval_cvlm.py):
#   --mixer_depth_map "cr:depth,..."  (env MIXER_DEPTH_MAP; включает миксер вместо ViT)
#   --mixed_cr_set "1 2 4 8 16"       (env MIXED_CR_SET; сэмплинг cr на батч, seed-детерминирован)

set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs

module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm
source /home/aburlova/cvlm/hse/protocol.env

ARM="${ARM:?set ARM=novit|fixed12direct|fixed12anchor|ratedepth}"
export MIXER_DEPTH_MAP="" VIT_INIT=pretrained FREEZE_VIT=0 NO_VIT=0
case "${ARM}" in
  novit)         export NO_VIT=1 ;;
  fixed12direct) export VIT_INIT=random ;;
  fixed12anchor) export MIXER_DEPTH_MAP="1:12,2:12,4:12,8:12,16:12" ;;
  ratedepth)     export MIXER_DEPTH_MAP="1:0,2:1,4:3,8:6,16:12" ;;
  *) echo "unknown ARM=${ARM}"; exit 1 ;;
esac
export MIXED_CR_SET="1 2 4 8 16"
# mixed-CR вместо куррикулума: расписание стадий выключаем.
export CR_SCHEDULE="" EVAL_CR_SCHEDULE=""
export COMPRESSION_RATE=1     # статический cr для фильтров датасета; сам cr батча сэмплируется
export EPOCHS=2               # ~7.4k шагов
export RUN_TAG="rd_${ARM}"
export OUTPUT_DIR="${RUNS_ROOT}/rd_${ARM}_${RD_STAMP:-$(date +%Y%m%d_%H%M%S)}"

echo "=== ARM=${ARM} MIXER_DEPTH_MAP=${MIXER_DEPTH_MAP} OUTPUT_DIR=${OUTPUT_DIR}"
cd "${CVLM_ROOT}"
bash scripts/run_full_pipeline.sh

# Пост-евал: качество финального чекпоинта на каждом cr из набора + невиданный 32
LAST=$(ls -1 "${OUTPUT_DIR}"/model_step_*.safetensors | sed -E 's|.*model_step_([0-9]+)\.safetensors|\1 &|' | sort -n | tail -1 | awk '{print $2}')
STEP=$(basename "${LAST}" | grep -oE '[0-9]+')
for CR in 1 2 4 8 16 32; do
  python src/eval_cvlm.py --mode cvlm \
    --checkpoint_path "${LAST}" --compression_rate "${CR}" \
    --dataset_name sggetao/PwC --dataset_split "${EVAL_SPLIT}" \
    --model_name_or_path "${MODEL_NAME}" --text_encoder_name "${TEXT_ENCODER_NAME}" \
    --vision_encoder_name "${VISION_ENCODER_NAME}" \
    --num_pool_latents "${NUM_POOL_LATENTS}" --max_prompt_len "${MAX_PROMPT_LEN}" \
    --max_answer_len "${MAX_ANSWER_LEN}" --max_vision_len "${MAX_VISION_LEN}" \
    --max_source_len "${MAX_SOURCE_LEN}" --max_samples "${EVAL_MAX_SAMPLES}" \
    --batch_size "${EVAL_BATCH_SIZE}" --seed "${SEED}" --global_step "${STEP}" \
    ${MIXER_DEPTH_MAP:+--mixer_depth_map "${MIXER_DEPTH_MAP}"} \
    --vit_init "${VIT_INIT}" --freeze_vit "${FREEZE_VIT}" --no_vit "${NO_VIT}" \
    --output_json "${OUTPUT_DIR}/eval_rd_cr${CR}.json" \
    --dump_per_sample "${OUTPUT_DIR}/eval_rd_cr${CR}_dump.jsonl" \
    --compute_generation_metrics
done
ls -1 "${OUTPUT_DIR}"/trainer_step_*.pt 2>/dev/null | sort -t_ -k3 -n | head -n -1 | xargs -r rm -f
echo "=== RD ARM DONE: ${ARM}"
