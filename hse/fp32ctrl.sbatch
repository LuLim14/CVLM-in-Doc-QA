#!/bin/bash
#SBATCH --job-name=cvlm_fp32
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
# РЕШАЮЩИЙ КОНТРОЛЬ ЧИСЛЕННОЙ ТОЧНОСТИ (fp32 master weights, scope=vision)
# ============================================================================
# ПРОБЛЕМА. Вся модель живёт в bf16, AdamW обновлял bf16-параметры НА МЕСТЕ
# (без autocast, без GradScaler, без мастер-весов). bf16 хранит 8 бит мантиссы,
# т.е. ОТНОСИТЕЛЬНАЯ гранулярность ~0.39-0.78% (2^-8..2^-7) от величины самого
# веса: ULP = 7.8e-3 при |w|~1.0 и ~1.2e-4 при |w|~0.02. Апдейт AdamW по модулю
# ~lr = 1e-4. Для КРУПНЫХ предобученных весов ViT это МЕНЬШЕ половины ULP =>
# округление стирает апдейт на каждом шаге; для мелких случайных весов — не
# стирает. Замер на прогоне rd5/rd6: 63.6% весов предобученного ViT ПОБИТОВО не
# изменились между шагом 1000 и 11326 (дрейф 2.77%) против 19.5% дрейфа у
# random-init арма. Значит и клейм «предобученный приор вредит», и ЦЕНТРАЛЬНЫЙ
# клейм «пост-ботлнек глубина вредит» конфаундены с численным артефактом,
# коллинеарным лечению.
#
# КОНТРОЛЬ. Те же армы, но FP32_MASTER_WEIGHTS=1 + FP32_MASTER_SCOPE=vision:
# fp32-мастера и fp32-моменты создаются ТОЛЬКО для ViT/mixer (~86M параметров).
# Декодер, текстовый энкодер, проекторы, pos_embed и pool_queries остаются на
# ПОБИТОВО прежнем bf16-пути. Это критично:
#   * научно — иначе прогон меняет численную оптимизацию ЧЕТЫРЁХ подмодулей
#     сразу и перестаёт быть тестом гипотезы «квантование апдейтов ViT
#     объясняет разрыв»;
#   * по памяти — ДЕКОДЕР ОБУЧАЕМЫЙ (реальный лог: trainable 1,835,876,608 из
#     1,874,560,000, 97.94%), scope=all стоил бы +20.5 GiB стейта оптимизатора
#     и почти наверняка OOM; scope=vision стоит +1.0 GiB.
#
# ПРОТОКОЛ — ПОЛНЫЙ ИСХОДНЫЙ, без сокращений: то же расписание cr, те же
# EPOCHS=3, тот же seed=42, те же батчи, что у legacy-армов (abl_vit /
# abl_randomvit). Любое укорочение сделало бы сравнение с ними невалидным.
#
# ДИАГНОСТИКА (лог + trackio + mechanistic.jsonl каждые LOG_INTERVAL шагов):
#   vit_frozen_frac=0.XXXX   доля весов ViT/mixer, ПОБИТОВО не изменившихся с
#                            прошлого лог-шага;
#   vit_drift=0.XXXX         относительный L2-дрейф ЖИВЫХ bf16-весов от W0;
#   vit_master_drift=0.XXXX  тот же дрейф, но по fp32-МАСТЕРАМ (честная мера
#                            того, куда реально ушёл оптимизатор);
#   vit_master_resid[_max]   ||master - bf16_live|| (норма отн. ||W0|| и max
#                            abs) — НАКОПЛЕННЫЙ, но ещё не проявившийся в bf16
#                            апдейт. В legacy-пути эта величина уничтожалась.
#   W0 берётся из mechanistic_init.safetensors (переживает resume).
#
# ЗАПУСК (ничего не запускается автоматически; порядок важен):
#   # 1) ПРИОРИТЕТНЫЙ арм — защищает ЦЕНТРАЛЬНЫЙ клейм о пост-ботлнек глубине:
#   sbatch --job-name=fp32_randomvit --export=ALL,ARM=randomvit fp32ctrl.sbatch
#   # 2) вторым — предобученный ViT (вторичный клейм про визуальный приор):
#   sbatch --job-name=fp32_vit       --export=ALL,ARM=vit       fp32ctrl.sbatch
# Армы независимы: каждый — своя джоба, свой OUTPUT_DIR, можно запускать по
# одному. Повторный запуск с тем же OUTPUT_DIR резюмит с последнего чекпоинта
# (FP32_MASTER_WEIGHTS/FP32_MASTER_SCOPE обязаны совпадать — иначе train падает
# с явной ошибкой, чтобы мастер-копии не потерялись молча).
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
# Код обязан УЖЕ содержать флаги: иначе молча получили бы обычный bf16-прогон.
grep -q -- "--fp32_master_scope" "${CVLM_ROOT}/src/train_cvlm.py" || {
  echo "ERROR: ${CVLM_ROOT}/src/train_cvlm.py has no --fp32_master_scope; sync the code first"
  exit 1
}
echo "=== CODE VERSION: $(cat "${CVLM_ROOT}/VERSION")"

# ARM по умолчанию — randomvit (приоритетный: центральный клейм).
ARM="${ARM:-randomvit}"
export VIT_INIT=pretrained FREEZE_VIT=0 NO_VIT=0 MIXER_DEPTH_MAP=""
case "${ARM}" in
  randomvit) export VIT_INIT=random ;;   # та же архитектура, config-init
  vit)       ;;                          # предобученный ViT-Base
  *) echo "unknown ARM=${ARM} (expected randomvit|vit)"; exit 1 ;;
esac

# --- ЭТО и есть лечение эксперимента --------------------------------------
export FP32_MASTER_WEIGHTS=1
export FP32_MASTER_SCOPE=vision   # ТОЛЬКО ViT/mixer; всё остальное — legacy

# --- ПОЛНЫЙ протокол, идентичный legacy-армам ------------------------------
# Значения дублируют protocol.env НАМЕРЕННО: сравнение идёт с уже отснятыми
# abl_vit/abl_randomvit, поэтому расписание пиннится здесь и не должно
# «поехать» вслед за будущими правками протокола.
export CR_SCHEDULE="1:1500,2:3000,4:4500,8:6000,16:0"
export EPOCHS=3                # как в основной кампании (~11.3k opt-шагов)
export COMPRESSION_RATE=16     # только для финального eval: последняя стадия
                               # (в трейне cr берётся из CR_SCHEDULE)
export SAVE_INTERVAL_STEPS=1000  # как в protocol.env: ckpt ~3.7 GB каждый
export LOG_INTERVAL=10         # окно vit_frozen_frac == LOG_INTERVAL шагов
export SEED=42                 # те же батчи, что у legacy-армов

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

# Resume после таймаута/падения (полный протокол ~9-10 ч трейна + eval; окно
# 24 ч, но повторный запуск всё равно полезен при перезапусках).
if compgen -G "${OUTPUT_DIR}/model_step_"*.safetensors >/dev/null 2>&1; then
  export RESUME_DIR="${OUTPUT_DIR}"
  echo "=== RESUME from ${OUTPUT_DIR} (FP32_MASTER_WEIGHTS=1, SCOPE=vision must stay)"
fi
if [ -f "${OUTPUT_DIR}/TRAIN_DONE" ]; then
  echo "=== TRAIN_DONE marker present; nothing to do"
  exit 0
fi

echo "=== node $(hostname), gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "=== ARM=${ARM} VIT_INIT=${VIT_INIT} FP32_MASTER_WEIGHTS=1 SCOPE=${FP32_MASTER_SCOPE}"
echo "=== CR_SCHEDULE=${CR_SCHEDULE} EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE} LR=${LR} SEED=${SEED}"
echo "=== OUTPUT_DIR=${OUTPUT_DIR}"
# Память: scope=vision добавляет ~1.0 GiB стейта оптимизатора (мастер + fp32
# grad + fp32 моменты на ~86M параметров ViT) + ~0.34 GiB на снапшоты
# диагностики. При OOM: export BATCH_SIZE=32 GRAD_ACCUM=2 (тот же эффективный
# батч 64) — но тогда ОБА арма, иначе сравнение поедет.

cd "${CVLM_ROOT}"
cp VERSION "${OUTPUT_DIR}/CODE_VERSION"
sha256sum src/*.py scripts/run_full_pipeline.sh hse/fp32ctrl.sbatch hse/protocol.env \
  > "${OUTPUT_DIR}/CODE_SHA256"
{
  echo "EXPERIMENT=fp32_master_weights_control"
  echo "ARM=${ARM}"
  echo "VIT_INIT=${VIT_INIT}"
  echo "FP32_MASTER_WEIGHTS=${FP32_MASTER_WEIGHTS}"
  echo "FP32_MASTER_SCOPE=${FP32_MASTER_SCOPE}"
  echo "CR_SCHEDULE=${CR_SCHEDULE}"
  echo "COMPRESSION_RATE_EVAL=${COMPRESSION_RATE}"
  echo "EPOCHS=${EPOCHS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "GRAD_ACCUM=${GRAD_ACCUM}"
  echo "LR=${LR}"
  echo "DISABLE_LR_SCHEDULE=${DISABLE_LR_SCHEDULE}"
  echo "ENABLE_WARMUP=${ENABLE_WARMUP} WARMUP_STEPS=${WARMUP_STEPS}"
  echo "UNFREEZE_ENCODER_TOP_K=${UNFREEZE_ENCODER_TOP_K}"
  echo "NUM_POOL_LATENTS=${NUM_POOL_LATENTS}"
  echo "MAX_SOURCE_LEN=${MAX_SOURCE_LEN}"
  echo "MAX_VISION_LEN=${MAX_VISION_LEN}"
  echo "SAVE_INTERVAL_STEPS=${SAVE_INTERVAL_STEPS}"
  echo "LOG_INTERVAL=${LOG_INTERVAL}"
  echo "SEED=${SEED}"
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

sha256sum -c "${OUTPUT_DIR}/CODE_SHA256"
ls -1 "${OUTPUT_DIR}"/trainer_step_*.pt 2>/dev/null | sort -t_ -k3 -n | head -n -1 | xargs -r rm -f
echo "=== FP32 CONTROL DONE: ARM=${ARM} -> ${OUTPUT_DIR}"
echo "=== next: grep -h 'vit_frozen_frac' ${OUTPUT_DIR}/pipeline.log | tail -20"
