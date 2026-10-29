#!/bin/bash
#SBATCH --job-name=cvlm_abl2
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# ============================================================================
# ДВЕ НЕДОСТАЮЩИЕ АБЛЯЦИИ (требование научного советника)
# ============================================================================
# ARM=meanpool — MeanPool+Project (P0)
#   ПРОБЛЕМА. Сильный арм (Pool+Project) использует ОБУЧАЕМЫЙ attention-пул с
#   K=4 латентными запросами. Проверено по чекпоинтам: pool_queries стартуют с
#   нуля (при нуле softmax равномерен => пул тождественен mean-pool), на шаге
#   1500 ещё ровно нули, но к финалу уезжают (норма 5.25). То есть «пул почти
#   средний» — это утверждение про ИНИЦИАЛИЗАЦИЮ, а не про обученную модель, и
#   рецензент вправе спросить, что даёт обучаемость пула.
#   АРМ. MEAN_POOL=1: пул — ТОЧНОЕ (побитово) среднее по РЕАЛЬНЫМ токенам чанка
#   на всей траектории. pool_queries остаются в модели (та же форма, тот же ключ
#   манифеста), но заморожены нулями: не получают градиента, не попадают в
#   оптимизатор и не сохраняются в чекпоинт. Геометрия остальной модели —
#   как у Pool+Project (NO_VIT=1), чтобы разница была ровно в одном факторе.
#
# ARM=mixer1l — Pool+1L (P1)
#   ПРОБЛЕМА. Есть только «0 слоёв против 12»; по двум точкам нельзя судить о
#   всей пост-ботлнек глубине (0 vs 12 может быть и порогом, и наклоном).
#   АРМ. VIT_INIT=random + VIT_NUM_LAYERS=1 — ОБЫЧНЫЙ (НЕ гейтированный) стек
#   ViTLayer, усечённый до ПЕРВОГО слоя. Это ровно тот же путь, что у
#   legacy-арма «Pool+12L Mixer» (VIT_INIT=random), и отличается от него ТОЛЬКО
#   числом слоёв: тот же тип блока и конфиг, тот же финальный LayerNorm, та же
#   маска паддинга, та же random-инициализация, те же общие веса и батчи
#   (shared_init_seed = seed+10000). Полный 12-слойный стек строится и потом
#   усекается, поэтому выжившие блоки побитово совпадают с первыми блоками
#   12-слойного арма, а RNG для общих модулей не сдвигается.
#   ПОЧЕМУ НЕ MIXER_DEPTH_MAP="1:1,...". Миксер zero-gated: его «глубина 1» на
#   старте тождественна 0 слоям, и арм измерял бы ДИНАМИКУ ГЕЙТА, а не глубину.
#   Для точки на кривой 0/1/12 нужен обычный слой (tests/test_meanpool_and_depth1.py).
#
# ЧТО НЕЛЬЗЯ ЗАБЫТЬ ПРО СРАВНИМОСТЬ (читать перед интерпретацией!):
#   * пара по умолчанию — legacy-армы seed 43: Pool+Project (NO_VIT=1, точка 0)
#     и Pool+12L Mixer (VIT_INIT=random, точка 12). Протокол здесь скопирован с
#     них (CR_SCHEDULE/EPOCHS/BATCH/LR/warmup/seed);
#   * MASK_VIT_PADDING по умолчанию 0 — как у legacy-армов (train_branch.sbatch
#     их не выставлял, дефолт пайплайна 0). Если парный legacy-прогон снимался с
#     1, выставить MASK_VIT_PADDING=1 ЗДЕСЬ (сверить с его EXPERIMENT_FLAGS),
#     иначе сравнение поедет на два фактора сразу;
#   * VIT_APPLY_FINAL_LAYERNORM=1 — тоже как у legacy-армов.
#
# ЗАПУСК (ничего не запускается автоматически):
#   sbatch --job-name=abl_meanpool --export=ALL,ARM=meanpool ablation_arms.sbatch
#   sbatch --job-name=abl_mixer1l  --export=ALL,ARM=mixer1l  ablation_arms.sbatch
# Сид: по умолчанию 43 (пара к готовым legacy-армам). Другой сид:
#   sbatch --export=ALL,ARM=meanpool,ABLATION_SEED=44 ablation_arms.sbatch
# Повторный запуск с тем же OUTPUT_DIR резюмит с последнего чекпоинта.
# ============================================================================

set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs

module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm

SNAPSHOT_ROOT="${CVLM_SNAPSHOT_ROOT:-/home/aburlova/cvlm}"
source "${SNAPSHOT_ROOT}/hse/protocol.env"
export CVLM_ROOT="${SNAPSHOT_ROOT}"
test -s "${CVLM_ROOT}/VERSION" || { echo "ERROR: missing ${CVLM_ROOT}/VERSION"; exit 1; }
echo "=== CODE VERSION: $(cat "${CVLM_ROOT}/VERSION")"

# Код обязан УЖЕ содержать оба флага в ОБОИХ скриптах И в пайплайне: иначе арм
# молча обучился/оценился бы другой архитектурой (ровно тот класс тихой подмены
# конфигурации, который уже один раз стоил нам прогона).
for _f in src/train_cvlm.py src/eval_cvlm.py scripts/run_full_pipeline.sh; do
  for _flag in --mean_pool --vit_num_layers; do
    grep -q -- "${_flag}" "${CVLM_ROOT}/${_f}" || {
      echo "ERROR: ${CVLM_ROOT}/${_f} has no ${_flag}; sync the code first"; exit 1; }
  done
done

# --- АРМ ------------------------------------------------------------------
# Явный case со ВСЕМИ армами и жёстким выходом на неизвестном значении: ARM,
# который молча даёт конфигурацию другого арма, уже однажды испортил прогон.
ARM="${ARM:?set ARM=meanpool|mixer1l}"
export MEAN_POOL=0
export VIT_NUM_LAYERS=0
export MIXER_DEPTH_MAP=""
export VIT_INIT=pretrained
export FREEZE_VIT=0
export NO_VIT=0
export MASK_VIT_PADDING="${MASK_VIT_PADDING:-0}"
export VIT_APPLY_FINAL_LAYERNORM=1
export COMPACT_DECODER_POSITIONS=0
case "${ARM}" in
  meanpool)
    # P0: пул = точное masked mean; всё остальное — как у Pool+Project (no-ViT).
    export MEAN_POOL=1
    export NO_VIT=1
    export VIT_NUM_LAYERS=0
    export MIXER_DEPTH_MAP=""
    ARM_EXPECT_LOG="MEAN-POOL ablation"
    ;;
  mixer1l)
    # P1: ОБЫЧНЫЙ (не гейтированный) random-init ViT, усечённый до 1 слоя.
    # Отличие от legacy-арма Pool+12L Mixer — ровно число слоёв.
    export MEAN_POOL=0
    export NO_VIT=0
    export VIT_INIT=random
    export VIT_NUM_LAYERS=1
    export MIXER_DEPTH_MAP=""
    ARM_EXPECT_LOG="VIT DEPTH ablation: vision encoder TRUNCATED to the first 1 of 12 layers"
    ;;
  *)
    echo "unknown ARM=${ARM} (expected meanpool|mixer1l)"; exit 1 ;;
esac

# --- ПОЛНЫЙ протокол, идентичный сид-репликациям ---------------------------
# Значения дублируют protocol.env НАМЕРЕННО: сравнение идёт с уже отснятыми
# legacy-армами, поэтому расписание пиннится здесь и не должно «поехать» вслед
# за будущими правками протокола. BATCH_SIZE/LR/warmup берутся из protocol.env.
export CR_SCHEDULE="1:1500,2:3000,4:4500,8:6000,16:0"
export MIXED_CR_SET=""         # куррикулум, а не mixed-cr
export EPOCHS=3
export COMPRESSION_RATE=16     # только для финального eval (headline cr=16);
                               # в трейне cr берётся из CR_SCHEDULE
export SAVE_INTERVAL_STEPS=1000
export LOG_INTERVAL=10
export NPROC=1                 # DDP не используем (см. protocol.env)
export SEED="${ABLATION_SEED:-43}"

# --- eval: только финальный чекпоинт, только cvlm --------------------------
export EVAL_ALL_CHECKPOINTS=0
export EVAL_MODES="cvlm"
export EVAL_COMPUTE_GEN=0
# Пер-чекпоинтная привязка cr работает ТОЛЬКО при EVAL_ALL_CHECKPOINTS=1, так
# что при 0 значение не используется. (Пустая строка всё равно подменяется
# дефолтом в run_full_pipeline.sh через ${VAR:-...} — это не баг здесь.)
export EVAL_CR_SCHEDULE=""

export RUN_TAG="abl2_${ARM}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/abl2_${ARM}_s${SEED}_${STAMP:-$(date +%Y%m%d_%H%M%S)}}"
mkdir -p "${OUTPUT_DIR}"

if compgen -G "${OUTPUT_DIR}/model_step_"*.safetensors >/dev/null 2>&1; then
  export RESUME_DIR="${OUTPUT_DIR}"
  echo "=== RESUME from ${OUTPUT_DIR} (ARM=${ARM} must stay identical)"
fi
if [ -f "${OUTPUT_DIR}/TRAIN_DONE" ]; then
  echo "=== TRAIN_DONE marker present; nothing to do"
  exit 0
fi

echo "=== node $(hostname), gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
# ФАКТИЧЕСКИЕ флаги (то, что реально уйдёт в train/eval) — одной строкой:
echo "=== ARM=${ARM} MEAN_POOL=${MEAN_POOL} VIT_NUM_LAYERS=${VIT_NUM_LAYERS} MIXER_DEPTH_MAP=${MIXER_DEPTH_MAP:-<off>} NO_VIT=${NO_VIT} VIT_INIT=${VIT_INIT} FREEZE_VIT=${FREEZE_VIT} MASK_VIT_PADDING=${MASK_VIT_PADDING} VIT_APPLY_FINAL_LAYERNORM=${VIT_APPLY_FINAL_LAYERNORM} NUM_POOL_LATENTS=${NUM_POOL_LATENTS}"
echo "=== CR_SCHEDULE=${CR_SCHEDULE} EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE} GRAD_ACCUM=${GRAD_ACCUM} LR=${LR} WARMUP=${ENABLE_WARMUP}/${WARMUP_STEPS} SEED=${SEED} NPROC=${NPROC}"
echo "=== EVAL_ALL_CHECKPOINTS=${EVAL_ALL_CHECKPOINTS} EVAL_MODES=${EVAL_MODES} EVAL_COMPUTE_GEN=${EVAL_COMPUTE_GEN} EVAL_CR_SCHEDULE='${EVAL_CR_SCHEDULE}' EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES}"
echo "=== OUTPUT_DIR=${OUTPUT_DIR}"

cd "${CVLM_ROOT}"
cp VERSION "${OUTPUT_DIR}/CODE_VERSION"
sha256sum src/*.py scripts/run_full_pipeline.sh hse/ablation_arms.sbatch hse/protocol.env \
  > "${OUTPUT_DIR}/CODE_SHA256"
{
  echo "EXPERIMENT=missing_ablation_arms"
  echo "ARM=${ARM}"
  echo "MEAN_POOL=${MEAN_POOL}"
  echo "VIT_NUM_LAYERS=${VIT_NUM_LAYERS}"
  echo "MIXER_DEPTH_MAP=${MIXER_DEPTH_MAP}"
  echo "NO_VIT=${NO_VIT}"
  echo "VIT_INIT=${VIT_INIT}"
  echo "FREEZE_VIT=${FREEZE_VIT}"
  echo "MASK_VIT_PADDING=${MASK_VIT_PADDING}"
  echo "VIT_APPLY_FINAL_LAYERNORM=${VIT_APPLY_FINAL_LAYERNORM}"
  echo "COMPACT_DECODER_POSITIONS=${COMPACT_DECODER_POSITIONS}"
  echo "NUM_POOL_LATENTS=${NUM_POOL_LATENTS}"
  echo "CR_SCHEDULE=${CR_SCHEDULE}"
  echo "COMPRESSION_RATE_EVAL=${COMPRESSION_RATE}"
  echo "EPOCHS=${EPOCHS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "GRAD_ACCUM=${GRAD_ACCUM}"
  echo "LR=${LR}"
  echo "DISABLE_LR_SCHEDULE=${DISABLE_LR_SCHEDULE}"
  echo "ENABLE_WARMUP=${ENABLE_WARMUP} WARMUP_STEPS=${WARMUP_STEPS}"
  echo "UNFREEZE_ENCODER_TOP_K=${UNFREEZE_ENCODER_TOP_K}"
  echo "MAX_SOURCE_LEN=${MAX_SOURCE_LEN}"
  echo "MAX_VISION_LEN=${MAX_VISION_LEN}"
  echo "SAVE_INTERVAL_STEPS=${SAVE_INTERVAL_STEPS}"
  echo "LOG_INTERVAL=${LOG_INTERVAL}"
  echo "NPROC=${NPROC}"
  echo "SEED=${SEED}"
  echo "EVAL_ALL_CHECKPOINTS=${EVAL_ALL_CHECKPOINTS}"
  echo "EVAL_MODES=${EVAL_MODES}"
  echo "EVAL_COMPUTE_GEN=${EVAL_COMPUTE_GEN}"
  echo "EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES}"
  echo "CODE_VERSION=$(cat VERSION)"
} > "${OUTPUT_DIR}/EXPERIMENT_FLAGS"

# Диск-гигиена: trainer_step_*.pt ~8 GB (моменты по ~1.84B обучаемых) — держим
# только самый свежий, он нужен лишь для resume.
(
  while true; do
    sleep 900
    ls -1 "${OUTPUT_DIR}"/trainer_step_*.pt 2>/dev/null | sort -t_ -k3 -n | head -n -1 | xargs -r rm -f
  done
) &
CLEANER_PID=$!
trap 'kill ${CLEANER_PID} 2>/dev/null || true' EXIT

bash scripts/run_full_pipeline.sh
touch "${OUTPUT_DIR}/TRAIN_DONE"

# ПОСТ-ПРОВЕРКА: модель сама печатает, какую архитектуру собрала. Если ожидаемой
# строки нет — арм собрался НЕ ТОТ, и результат нельзя класть в таблицу.
if ! grep -qF "${ARM_EXPECT_LOG}" "${OUTPUT_DIR}/pipeline.log"; then
  echo "ERROR: ARM=${ARM} expected marker not found in pipeline.log: ${ARM_EXPECT_LOG}"
  echo "       the run built a DIFFERENT architecture than requested — do not report it"
  exit 1
fi
echo "=== ARM MARKER OK: ${ARM_EXPECT_LOG}"

sha256sum -c "${OUTPUT_DIR}/CODE_SHA256"
ls -1 "${OUTPUT_DIR}"/trainer_step_*.pt 2>/dev/null | sort -t_ -k3 -n | head -n -1 | xargs -r rm -f
echo "=== ABLATION ARM DONE: ${ARM} (seed=${SEED}) -> ${OUTPUT_DIR}"
echo "=== next: cat ${OUTPUT_DIR}/eval_cvlm.json  # pair with the seed-${SEED} legacy arms"
