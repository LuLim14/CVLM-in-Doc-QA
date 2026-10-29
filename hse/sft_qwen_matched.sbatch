#!/bin/bash
#SBATCH --job-name=qwen_sft
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Matched first-N reader for the existing Qwen2.5-0.5B Pool+Project result.
# Submit one independent job per rate:
#   sbatch --job-name=qwen_sft_cr8  --export=ALL,CR=8  sft_qwen_matched.sbatch
#   sbatch --job-name=qwen_sft_cr16 --export=ALL,CR=16 sft_qwen_matched.sbatch

set -euo pipefail

module purge
module load python/miniconda cuda/12.9.1
source activate /home/aburlova/conda_envs/cvlm

SNAP="${CVLM_SNAPSHOT_ROOT:-/home/aburlova/cvlm_reviewfix}"
source "$SNAP/hse/protocol.env"
unset EVAL_ALL_CHECKPOINTS EVAL_CR_SCHEDULE CR_SCHEDULE

CR="${CR:?set CR=8 or CR=16}"
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${RUNS_ROOT}/qwen_sft_firstn_cr${CR}_${STAMP}"
mkdir -p "$OUT" /home/aburlova/cvlm/runs/_logs

cd "$SNAP"
export PYTHONPATH="$SNAP/src"
export PYTHONUNBUFFERED=1

echo "=== QWEN MATCHED-SFT CR=$CR MODEL=$QWEN_MODEL_NAME SNAP=$SNAP OUT=$OUT"
python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$QWEN_MODEL_NAME', local_files_only=True); print('QWEN_CACHE_OK')"
sha256sum src/train_sft.py src/eval_cvlm.py hse/protocol.env hse/sft_qwen_matched.sbatch \
  > "$OUT/src_hashes.txt"

python src/train_sft.py \
  --output_dir "$OUT" \
  --dataset_name sggetao/PwC \
  --dataset_split train \
  --model_name_or_path "$QWEN_MODEL_NAME" \
  --max_prompt_len 512 \
  --max_answer_len 1024 \
  --truncate_cr "$CR" \
  --max_vision_len 1024 \
  --max_source_len 1024 \
  --text_encoder_name answerdotai/ModernBERT-base \
  --epochs 3 \
  --batch_size 64 \
  --gradient_accumulation_steps 1 \
  --lr 1e-4 \
  --lr_scheduler_type constant_with_warmup \
  --enable_warmup \
  --warmup_steps 100 \
  --grad_clip 1.0 \
  --gradient_checkpointing \
  --log_interval 10 \
  --save_interval_steps 1000 \
  --seed 42 \
  2>&1 | tee "$OUT/pipeline.log"

touch "$OUT/TRAIN_DONE"
EOUT="$OUT/eval_fresh"
mkdir -p "$EOUT"
python src/eval_cvlm.py \
  --mode sft \
  --sft_model_path "$OUT" \
  --dataset_name sggetao/PwC \
  --dataset_split "test[2100:5100]" \
  --model_name_or_path "$QWEN_MODEL_NAME" \
  --text_encoder_name answerdotai/ModernBERT-base \
  --vision_encoder_name google/vit-base-patch16-224 \
  --num_pool_latents 4 \
  --max_prompt_len 512 \
  --max_answer_len 1024 \
  --max_vision_len 1024 \
  --max_source_len 1024 \
  --max_samples 0 \
  --batch_size 8 \
  --seed 42 \
  --compression_rate 1 \
  --doc_truncation first \
  --doc_budget_cr "$CR" \
  --output_json "$EOUT/qwen_sft_firstn_cr${CR}.json" \
  --dump_per_sample "$EOUT/qwen_sft_firstn_cr${CR}_dump.jsonl" \
  2>&1 | tee "$OUT/eval_fresh.log"

touch "$OUT/DONE"
echo "=== QWEN_MATCHED_SFT_DONE CR=$CR OUT=$OUT"
