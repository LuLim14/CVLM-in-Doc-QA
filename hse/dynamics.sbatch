#!/bin/bash
#SBATCH --job-name=cvlm_dyn
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Диагностика optimization dynamics по рецепту советника (для Figure 1 / RQ1):
#   - фиксированный cr=1 (без смены стадий),
#   - warmup пропорциональный (20 из 250 шагов),
#   - 3 seeds x 4 ветки, битово одинаковые общие модули (shared_init_seed)
#     и детерминированный порядок батчей (DataLoader generator),
#   - чекпоинты каждые 50 шагов, TF-eval на 1000 сэмплах с пер-сэмпловым
#     дампом (paired bootstrap между ветками на одном seed).
# Требует кода с фиксом RNG (init-hash печатается в лог — сверить между ветками!).

set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs

module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm
source /home/aburlova/cvlm/hse/protocol.env

cd "${CVLM_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"

for SEED_V in 42 43 44; do
  for BR in vit randomvit novit frozenvit; do
    export VIT_INIT=pretrained FREEZE_VIT=0 NO_VIT=0
    case "${BR}" in
      randomvit) export VIT_INIT=random ;;
      novit)     export NO_VIT=1 ;;
      frozenvit) export FREEZE_VIT=1 ;;
    esac
    export RUN_TAG="dyn_${BR}_s${SEED_V}"
    export OUTPUT_DIR="${RUNS_ROOT}/dynamics_${STAMP}/s${SEED_V}/${BR}"
    echo; echo "########## DYNAMICS seed=${SEED_V} branch=${BR} ##########"
    SEED="${SEED_V}" MAX_SAMPLES=16000 EPOCHS=1 \
      CR_SCHEDULE="1:0" EVAL_CR_SCHEDULE="1:0" COMPRESSION_RATE=1 \
      WARMUP_STEPS=20 SAVE_INTERVAL_STEPS=50 \
      EVAL_MAX_SAMPLES=1000 EVAL_MODES="cvlm" EVAL_COMPUTE_GEN=0 EVAL_DUMP=1 \
      bash scripts/run_full_pipeline.sh
    grep -m1 'init-hash' "${OUTPUT_DIR}/pipeline.log" || true
  done
done

echo; echo "=== init-hash сверка (по каждому seed должны совпадать у всех 4 веток):"
grep -h 'init-hash' "${RUNS_ROOT}/dynamics_${STAMP}"/s*/*/pipeline.log | sort | uniq -c
echo "=== DYNAMICS DONE: ${RUNS_ROOT}/dynamics_${STAMP}"
