#!/bin/bash
#SBATCH --job-name=cvlm_fp32
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# ============================================================================
# РЕШАЮЩИЙ КОНТРОЛЬ ЧИСЛЕННОЙ ТОЧНОСТИ (fp32 master weights)
# ============================================================================
# ПРОБЛЕМА. Вся модель живёт в bf16, AdamW обновлял bf16-параметры НА МЕСТЕ
# (без autocast, без GradScaler, без мастер-весов). У bf16 8 бит мантиссы:
# ULP веса величины ~1.0 равен 2^-7 = 7.8e-3, веса величины ~0.02 — ~1.2e-4.
# Апдейт AdamW по модулю ~lr = 1e-4. Для КРУПНЫХ предобученных весов ViT это
# МЕНЬШЕ половины ULP => округление стирает апдейт на каждом шаге; для мелких
# случайных весов — не стирает. Замер на прогоне rd5/rd6: 63.6% весов
# предобученного ViT ПОБИТОВО не изменились между шагом 1000 и 11326 (дрейф
# 2.77%) против 19.5% дрейфа у random-init арма. Значит клейм «предобученный
# визуальный приор вредит» КОНФАУНДЕН с чисто численным артефактом, причём
# конфаундер КОЛЛИНЕАРЕН лечению (pretrained <=> крупные веса <=> крупный ULP).
#
# КОНТРОЛЬ. Те же два арма, но с FP32_MASTER_WEIGHTS=1: параметры остаются
# bf16 для forward/backward, а AdamW ведёт fp32-мастер-копии, считает моменты
# в fp32 и округляет обратно в bf16 после шага. Суб-ULP апдейты НАКАПЛИВАЮТСЯ,
# а не стираются. Если разрыв vit-vs-randomvit сохраняется — эффект настоящий;
# если исчезает/переворачивается — это была арифметика.
#
# ДИАГНОСТИКА (пишется в лог/CSV/mechanistic.jsonl каждые LOG_INTERVAL шагов):
#   vit_frozen_frac=0.XXXX  доля весов ViT/mixer, ПОБИТОВО не изменившихся с
#                           прошлого лог-шага;
#   vit_drift=0.XXXX        относительный L2-дрейф от инициализации
#                           (||W-W0||/||W0||, W0 из mechanistic_init.safetensors).
#
# ЗАПУСК (два арма — две джобы; ничего не запускается автоматически):
#   STAMP=$(date +%Y%m%d_%H%M%S)
#   sbatch --job-name=fp32_vit       --export=ALL,ARM=vit,STAMP=${STAMP}       fp32ctrl.sbatch
#   sbatch --job-name=fp32_randomvit --export=ALL,ARM=randomvit,STAMP=${STAMP} fp32ctrl.sbatch
# Повторный запуск с тем же OUTPUT_DIR резюмит с последнего чекпоинта
# (FP32_MASTER_WEIGHTS=1 обязателен и при resume — иначе train падает с явной
# ошибкой, чтобы мастер-копии не потерялись молча).
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
# Код обязан УЖЕ содержать флаг: иначе молча получили бы обычный bf16-прогон.
grep -q -- "--fp32_master_weights" "${CVLM_ROOT}/src/train_cvlm.py" || {
  echo "ERROR: ${CVLM_ROOT}/src/train_cvlm.py has no --fp32_master_weights; sync the code first"
  exit 1
}
echo "=== CODE VERSION: $(cat "${CVLM_ROOT}/VERSION")"

ARM="${ARM:?set ARM=vit|randomvit}"
export VIT_INIT=pretrained FREEZE_VIT=0 NO_VIT=0 MIXER_DEPTH_MAP=""
case "${ARM}" in
  vit)       ;;                          # предобученный ViT-Base (лечение)
  randomvit) export VIT_INIT=random ;;   # та же архитектура, config-init
  *) echo "unknown ARM=${ARM} (expected vit|randomvit)"; exit 1 ;;
esac

# --- ЭТО и есть лечение эксперимента --------------------------------------
export FP32_MASTER_WEIGHTS=1

# --- укороченный протокол: 4 стадии, решающая точка — шаг 3000 -------------
# Всё остальное (модели, lr=1e-4, batch, warmup, MAX_*_LEN, unfreeze, seed)
# берётся из protocol.env, чтобы контроль отличался от основных армов ТОЛЬКО
# длиной куррикулума и мастер-весами.
export CR_SCHEDULE="1:1000,2:2000,4:3000,8:0"
export COMPRESSION_RATE=8      # только для финального eval: последняя стадия
                               # куррикулума (в трейне cr берётся из CR_SCHEDULE)
export EPOCHS=1                # ~3.8k optimizer-шагов при bs=64 (эпоха PwC),
                               # т.е. все 4 стадии проходятся полностью;
                               # принудительные сейвы на границах 1000/2000/3000
export SAVE_INTERVAL_STEPS=500 # плотная сетка чекпоинтов для анализа дрейфа
export LOG_INTERVAL=10         # окно vit_frozen_frac == LOG_INTERVAL шагов
export SEED=42

# --- eval: только финальный чекпоинт, только cvlm --------------------------
export EVAL_ALL_CHECKPOINTS=0
export EVAL_MODES="cvlm"
export EVAL_COMPUTE_GEN=1
# EVAL_CR_SCHEDULE не нужен: пер-чекпоинтная привязка cr работает ТОЛЬКО при
# EVAL_ALL_CHECKPOINTS=1. (Пустое значение всё равно подменяется дефолтом в
# run_full_pipeline.sh через ${VAR:-...}, но при 0 оно нигде не используется.)
export EVAL_CR_SCHEDULE=""

export RUN_TAG="fp32_${ARM}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/fp32_${ARM}_${STAMP:-$(date +%Y%m%d_%H%M%S)}}"
mkdir -p "${OUTPUT_DIR}"

# Resume после таймаута/падения (12h окна должно хватать: ~4.5h трейн + eval).
if compgen -G "${OUTPUT_DIR}/model_step_"*.safetensors >/dev/null 2>&1; then
  export RESUME_DIR="${OUTPUT_DIR}"
  echo "=== RESUME from ${OUTPUT_DIR} (FP32_MASTER_WEIGHTS must stay 1)"
fi
if [ -f "${OUTPUT_DIR}/TRAIN_DONE" ]; then
  echo "=== TRAIN_DONE marker present; nothing to do"
  exit 0
fi

echo "=== node $(hostname), gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "=== ARM=${ARM} VIT_INIT=${VIT_INIT} FP32_MASTER_WEIGHTS=${FP32_MASTER_WEIGHTS}"
echo "=== CR_SCHEDULE=${CR_SCHEDULE} EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE} LR=${LR}"
echo "=== OUTPUT_DIR=${OUTPUT_DIR}"
# При OOM (fp32-мастера + fp32-моменты стоят ~+2.5 GB состояния оптимизатора):
#   export BATCH_SIZE=32 GRAD_ACCUM=2   -> тот же эффективный батч 64, ОБА арма.

cd "${CVLM_ROOT}"
cp VERSION "${OUTPUT_DIR}/CODE_VERSION"
sha256sum src/*.py scripts/run_full_pipeline.sh hse/fp32ctrl.sbatch hse/protocol.env \
  > "${OUTPUT_DIR}/CODE_SHA256"
{
  echo "EXPERIMENT=fp32_master_weights_control"
  echo "ARM=${ARM}"
  echo "VIT_INIT=${VIT_INIT}"
  echo "FP32_MASTER_WEIGHTS=${FP32_MASTER_WEIGHTS}"
  echo "CR_SCHEDULE=${CR_SCHEDULE}"
  echo "COMPRESSION_RATE_EVAL=${COMPRESSION_RATE}"
  echo "EPOCHS=${EPOCHS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "GRAD_ACCUM=${GRAD_ACCUM}"
  echo "LR=${LR}"
  echo "DISABLE_LR_SCHEDULE=${DISABLE_LR_SCHEDULE}"
  echo "UNFREEZE_ENCODER_TOP_K=${UNFREEZE_ENCODER_TOP_K}"
  echo "NUM_POOL_LATENTS=${NUM_POOL_LATENTS}"
  echo "MAX_SOURCE_LEN=${MAX_SOURCE_LEN}"
  echo "MAX_VISION_LEN=${MAX_VISION_LEN}"
  echo "SAVE_INTERVAL_STEPS=${SAVE_INTERVAL_STEPS}"
  echo "LOG_INTERVAL=${LOG_INTERVAL}"
  echo "SEED=${SEED}"
  echo "CODE_VERSION=$(cat VERSION)"
} > "${OUTPUT_DIR}/EXPERIMENT_FLAGS"

# Диск-гигиена: с fp32-мастерами trainer_step_*.pt весит ~2.6 GB — держим
# только самый свежий (нужен лишь для resume).
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

sha256sum -c "${OUTPUT_DIR}/CODE_SHA256"
ls -1 "${OUTPUT_DIR}"/trainer_step_*.pt 2>/dev/null | sort -t_ -k3 -n | head -n -1 | xargs -r rm -f
echo "=== FP32 CONTROL DONE: ARM=${ARM} -> ${OUTPUT_DIR}"
echo "=== next: grep -h 'vit_frozen_frac' ${OUTPUT_DIR}/pipeline.log | tail -20"
