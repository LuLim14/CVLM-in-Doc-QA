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
#   ARM=fixed12direct  : VIT_INIT=random (12 mask-safe слоёв БЕЗ гейтов и БЕЗ
#                        final LN, чтобы блоки совпадали с bare mixer stack)
#   ARM=fixed12unmasked: тот же direct-контроль, но legacy attention без padding
#                        mask — причинный контроль для Bottleneck-Safe Mixer
#   ARM=fixed12anchor  : MIXER_DEPTH_MAP="1:12,2:12,4:12,8:12,16:12" (zero-gated residual)
#   ARM=ratedepth      : MIXER_DEPTH_MAP="1:0,2:1,4:3,8:6,16:12"    (условная глубина)
# Пары сравнения: direct-vs-anchor использует одинаковые 12 ViTLayer blocks
# без финального LayerNorm и изолирует function-preserving insertion;
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
SNAPSHOT_ROOT="${CVLM_SNAPSHOT_ROOT:-/home/aburlova/cvlm_rd5}"
source "${SNAPSHOT_ROOT}/hse/protocol.env"
# ИЗОЛЯЦИЯ КАМПАНИИ: RD живёт в замороженном снапшоте, боевую копию не трогаем.
export CVLM_ROOT="${SNAPSHOT_ROOT}"
test -s "${CVLM_ROOT}/VERSION" || { echo "ERROR: missing ${CVLM_ROOT}/VERSION"; exit 1; }
echo "=== CODE VERSION: $(cat "${CVLM_ROOT}/VERSION")"
SMOKE="${SMOKE:-0}"
if [ "${SMOKE}" = "1" ]; then
  export MAX_SAMPLES=8000 EPOCHS=1 EVAL_MAX_SAMPLES=64
fi

ARM="${ARM:?set ARM=novit|fixed12direct|fixed12unmasked|fixed12anchor|ratedepth}"
export MIXER_DEPTH_MAP="" VIT_INIT=pretrained FREEZE_VIT=0 NO_VIT=0
export MASK_VIT_PADDING=1 VIT_APPLY_FINAL_LAYERNORM=1
export COMPACT_DECODER_POSITIONS=0
case "${ARM}" in
  novit)         export NO_VIT=1 ;;
  fixed12direct) export VIT_INIT=random VIT_APPLY_FINAL_LAYERNORM=0 ;;
  fixed12unmasked)
    export VIT_INIT=random VIT_APPLY_FINAL_LAYERNORM=0 MASK_VIT_PADDING=0
    ;;
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
# Avoid the generic all-checkpoint x all-baseline sweep. The final cross-CR
# paired eval below is the RD result; the pipeline only does one cheap sanity
# pass on the final checkpoint.
export EVAL_ALL_CHECKPOINTS=0 EVAL_MODES="cvlm" EVAL_COMPUTE_GEN=0
RD_COMPUTE_GEN="${RD_COMPUTE_GEN:-1}"

echo "=== ARM=${ARM} MIXER_DEPTH_MAP=${MIXER_DEPTH_MAP} OUTPUT_DIR=${OUTPUT_DIR}"
cd "${CVLM_ROOT}"
mkdir -p "${OUTPUT_DIR}"
cp VERSION "${OUTPUT_DIR}/CODE_VERSION"
sha256sum \
  src/modeling.py src/train_cvlm.py src/train_utils.py src/train_logging.py src/eval_cvlm.py \
  tools/measure_padding_contamination.py tools/block_bootstrap_train.py \
  scripts/run_full_pipeline.sh hse/ratedepth.sbatch hse/protocol.env \
  > "${OUTPUT_DIR}/CODE_SHA256"
{
  echo "ARM=${ARM}"
  echo "MIXER_DEPTH_MAP=${MIXER_DEPTH_MAP}"
  echo "MIXED_CR_SET=${MIXED_CR_SET}"
  echo "MASK_VIT_PADDING=${MASK_VIT_PADDING}"
  echo "VIT_APPLY_FINAL_LAYERNORM=${VIT_APPLY_FINAL_LAYERNORM}"
  echo "COMPACT_DECODER_POSITIONS=${COMPACT_DECODER_POSITIONS}"
  echo "GATE_LR_MULT=${GATE_LR_MULT:-1.0}"
  echo "MECHANISTIC_PROBE_SAMPLES=${MECHANISTIC_PROBE_SAMPLES:-16}"
  echo "SAVE_MECHANISTIC_INIT=${SAVE_MECHANISTIC_INIT:-1}"
  echo "SAVE_INTERVAL_STEPS=${SAVE_INTERVAL_STEPS}"
  echo "SEED=${SEED}"
} > "${OUTPUT_DIR}/EXPERIMENT_FLAGS"
bash scripts/run_full_pipeline.sh

# Пост-евал: качество финального чекпоинта на каждом cr из набора + невиданный 32
LAST=$(ls -1 "${OUTPUT_DIR}"/model_step_*.safetensors | sed -E 's|.*model_step_([0-9]+)\.safetensors|\1 &|' | sort -n | tail -1 | awk '{print $2}')
STEP=$(basename "${LAST}" | grep -oE '[0-9]+')
for CR in 1 2 4 8 16 32; do
  EVAL_ARGS=(
    src/eval_cvlm.py --mode cvlm
    --checkpoint_path "${LAST}" --compression_rate "${CR}" \
    --dataset_name sggetao/PwC --dataset_split "${EVAL_SPLIT}" \
    --model_name_or_path "${MODEL_NAME}" --text_encoder_name "${TEXT_ENCODER_NAME}" \
    --vision_encoder_name "${VISION_ENCODER_NAME}" \
    --num_pool_latents "${NUM_POOL_LATENTS}" --max_prompt_len "${MAX_PROMPT_LEN}" \
    --max_answer_len "${MAX_ANSWER_LEN}" --max_vision_len "${MAX_VISION_LEN}" \
    --max_source_len "${MAX_SOURCE_LEN}" --max_samples "${EVAL_MAX_SAMPLES}" \
    --batch_size "${EVAL_BATCH_SIZE}" --seed "${SEED}" --global_step "${STEP}" \
    --vit_init "${VIT_INIT}" --freeze_vit "${FREEZE_VIT}" --no_vit "${NO_VIT}" \
    --mask_vit_padding "${MASK_VIT_PADDING}" \
    --vit_apply_final_layernorm "${VIT_APPLY_FINAL_LAYERNORM}" \
    --compact_decoder_positions "${COMPACT_DECODER_POSITIONS}" \
    --output_json "${OUTPUT_DIR}/eval_rd_cr${CR}.json" \
    --dump_per_sample "${OUTPUT_DIR}/eval_rd_cr${CR}_dump.jsonl" \
  )
  if [[ -n "${MIXER_DEPTH_MAP}" ]]; then
    EVAL_ARGS+=( --mixer_depth_map "${MIXER_DEPTH_MAP}" )
  fi
  if [[ "${RD_COMPUTE_GEN}" == "1" ]]; then
    EVAL_ARGS+=( --compute_generation_metrics )
  fi
  python "${EVAL_ARGS[@]}"
  # Парный zero-gate прогон: если метрики НЕ ухудшаются при занулённых гейтах,
  # модель фактически не пользуется добавленной глубиной (диагностика, не метод).
  if [[ -n "${MIXER_DEPTH_MAP}" ]]; then
    ZERO_ARGS=( "${EVAL_ARGS[@]}" )
    for i in "${!ZERO_ARGS[@]}"; do
      ZERO_ARGS[$i]="${ZERO_ARGS[$i]/eval_rd_cr${CR}.json/eval_rd_cr${CR}_zerogate.json}"
      ZERO_ARGS[$i]="${ZERO_ARGS[$i]/eval_rd_cr${CR}_dump.jsonl/eval_rd_cr${CR}_zerogate_dump.jsonl}"
    done
    ZERO_ARGS+=( --force_zero_gates )
    python "${ZERO_ARGS[@]}"
  fi
done
sha256sum -c "${OUTPUT_DIR}/CODE_SHA256"
ls -1 "${OUTPUT_DIR}"/trainer_step_*.pt 2>/dev/null | sort -t_ -k3 -n | head -n -1 | xargs -r rm -f
echo "=== RD ARM DONE: ${ARM}"
