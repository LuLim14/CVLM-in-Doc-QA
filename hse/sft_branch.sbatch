#!/bin/bash
#SBATCH --job-name=cvlm_sft
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# SFT-бейзлайны. Параметр — env TRUNC_CR (через sbatch --export):
#   TRUNC_CR=0  : full-doc SFT (ceiling)
#   TRUNC_CR=8  : truncated-SFT при бюджете CVLM cr=8  (док -> первые ~N/8 токенов)
#   TRUNC_CR=16 : truncated-SFT при бюджете cr=16
# Пример: sbatch --job-name=sft_trunc8 --export=ALL,TRUNC_CR=8 sft_branch.sbatch

set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs

module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm

source /home/aburlova/cvlm/hse/protocol.env

TRUNC_CR="${TRUNC_CR:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TAG="sft_full"; [ "${TRUNC_CR}" != "0" ] && TAG="sft_trunc${TRUNC_CR}"
OUT="${RUNS_ROOT}/${TAG}_${STAMP}"
mkdir -p "${OUT}"

echo "=== node $(hostname), TRUNC_CR=${TRUNC_CR}, OUT=${OUT}"
cd "${CVLM_ROOT}"
export PYTHONPATH="${CVLM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

python src/train_sft.py \
  --output_dir "${OUT}" \
  --dataset_name sggetao/PwC \
  --dataset_split train \
  --model_name_or_path "${MODEL_NAME}" \
  --max_prompt_len "${MAX_PROMPT_LEN}" \
  --max_answer_len "${MAX_ANSWER_LEN}" \
  --truncate_cr "${TRUNC_CR}" \
  --max_vision_len "${MAX_VISION_LEN}" \
  --max_source_len "${MAX_SOURCE_LEN}" \
  --text_encoder_name "${TEXT_ENCODER_NAME}" \
  --epochs 2 \
  --batch_size "${BATCH_SIZE}" \
  --lr 1e-5 \
  --gradient_checkpointing \
  --save_interval_steps 1000 \
  --seed "${SEED}" \
  2>&1 | tee "${OUT}/pipeline.log"

echo "=== DONE ${TAG}"
