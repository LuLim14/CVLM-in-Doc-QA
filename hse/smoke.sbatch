#!/bin/bash
#SBATCH --job-name=cvlm_smoke
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Смоук перед массовым запуском (ОБЯЗАТЕЛЬНО прогнать первым):
#  1) все 4 ветки поднимаются, лосс падает, чекпоинт сохраняется и грузится в eval;
#  2) замер сек/шаг на стадии cr=1 при боевом BATCH_SIZE -> калибровка CR_SCHEDULE;
#  3) память: не OOM ли bs=64 на A100 80GB.
# Каждая ветка: ~60 оптимизаторных шагов на урезанном датасете + микро-eval.

set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs

module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm

source /home/aburlova/cvlm/hse/protocol.env

echo "=== node $(hostname), gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
cd "${CVLM_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
for BR in vit randomvit novit frozenvit; do
  export VIT_INIT=pretrained FREEZE_VIT=0 NO_VIT=0
  case "${BR}" in
    randomvit) export VIT_INIT=random ;;
    novit)     export NO_VIT=1 ;;
    frozenvit) export FREEZE_VIT=1 ;;
  esac
  export RUN_TAG="smoke_${BR}"
  export OUTPUT_DIR="${RUNS_ROOT}/smoke_${STAMP}/${BR}"
  echo; echo "########## SMOKE BRANCH: ${BR} ##########"
  MAX_SAMPLES=8000 EPOCHS=1 \
    CR_SCHEDULE="1:30,2:0" EVAL_CR_SCHEDULE="1:30,2:0" \
    EVAL_MAX_SAMPLES=64 EVAL_MODES="cvlm" EVAL_COMPUTE_GEN=1 \
    bash scripts/run_full_pipeline.sh
  # сек/шаг для калибровки: последние строки metrics.csv (колонка batch_time)
  echo "--- ${BR} batch_time tail:"; tail -5 "${OUTPUT_DIR}/metrics.csv" || true
done

echo; echo "=== SMOKE OK: все 4 ветки прошли train->save->eval. Проверь batch_time и nvidia-smi пики выше."
