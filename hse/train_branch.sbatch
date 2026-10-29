#!/bin/bash
#SBATCH --job-name=cvlm_abl
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Одна ветка абляции CVLM. rocky + --constraint=type_e (узлы с A100) +
# НЕтипизированный --gres=gpu:1: после переезда proj_1703->proj_1887 запросы
# типа gpu:a100:N отклоняются — HSE требует выбирать ТИП УЗЛА, не тип GPU
# (подтверждено поддержкой 29.07.2026; hpc.hse.ru/instructions/run#nodetype).
# Параметры через sbatch --export:
#   BRANCH=vit|randomvit|novit|frozenvit   (обязательно)
#   OUTPUT_DIR=/abs/path                    (опционально; launch_all задаёт заранее,
#                                            чтобы chained-resume job знал путь)
# Если в OUTPUT_DIR уже есть чекпоинты — автоматически продолжаем (resume),
# поэтому один и тот же sbatch годится и как первичная джоба, и как продолжение
# по --dependency=afterany.

set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs

module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm

source /home/aburlova/cvlm/hse/protocol.env

BRANCH="${BRANCH:?set BRANCH=vit|randomvit|novit|frozenvit}"
export VIT_INIT=pretrained FREEZE_VIT=0 NO_VIT=0
case "${BRANCH}" in
  vit)       ;;
  randomvit) export VIT_INIT=random ;;
  novit)     export NO_VIT=1 ;;
  frozenvit) export FREEZE_VIT=1 ;;
  *) echo "unknown BRANCH=${BRANCH}"; exit 1 ;;
esac

export RUN_TAG="abl_${BRANCH}"
if [ -z "${OUTPUT_DIR:-}" ]; then
  export OUTPUT_DIR="${RUNS_ROOT}/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUTPUT_DIR}"
# Продолжение после таймаута/падения: если чекпоинты уже есть — резюмим.
if compgen -G "${OUTPUT_DIR}/model_step_"*.safetensors >/dev/null 2>&1; then
  export RESUME_DIR="${OUTPUT_DIR}"
  echo "=== RESUME from ${OUTPUT_DIR}"
fi
# Если тренировка уже добежала (маркер) — сразу к eval/выходу.
if [ -f "${OUTPUT_DIR}/TRAIN_DONE" ]; then
  echo "=== TRAIN_DONE marker present; nothing to do (chained job after clean finish)"
  exit 0
fi

echo "=== node $(hostname), gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "=== BRANCH=${BRANCH} VIT_INIT=${VIT_INIT} FREEZE_VIT=${FREEZE_VIT} NO_VIT=${NO_VIT}"
echo "=== OUTPUT_DIR=${OUTPUT_DIR}"

# Диск-гигиена на лету: optimizer-стейты ~8GB/шт нужны только для resume —
# фоновый чистильщик держит только самый свежий trainer_step_*.pt.
(
  while true; do
    sleep 900
    ls -1 "${OUTPUT_DIR}"/trainer_step_*.pt 2>/dev/null | sort -t_ -k3 -n | head -n -1 | xargs -r rm -f
  done
) &
CLEANER_PID=$!
trap 'kill ${CLEANER_PID} 2>/dev/null || true' EXIT

cd "${CVLM_ROOT}"
bash scripts/run_full_pipeline.sh
touch "${OUTPUT_DIR}/TRAIN_DONE"

ls -1 "${OUTPUT_DIR}"/trainer_step_*.pt 2>/dev/null | sort -t_ -k3 -n | head -n -1 | xargs -r rm -v
echo "=== DONE ${BRANCH}"
