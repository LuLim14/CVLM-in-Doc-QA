#!/bin/bash
#SBATCH --job-name=selected_sft
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Distribution-matched reader for an offline selection policy. Required env:
# SELECTOR=bm25|llmlingua, CR=8|16, TRAIN_SELECTION_FILE, EVAL_SELECTION_FILE.

set -euo pipefail

module purge
module load python/miniconda cuda/12.9.1
source activate /home/aburlova/conda_envs/cvlm

SNAP="${CVLM_SNAPSHOT_ROOT:-/home/aburlova/cvlm_reviewfix}"
source "$SNAP/hse/protocol.env"
unset EVAL_ALL_CHECKPOINTS EVAL_CR_SCHEDULE CR_SCHEDULE

SELECTOR="${SELECTOR:?set SELECTOR=bm25 or llmlingua}"
CR="${CR:?set CR=8 or CR=16}"
TRAIN_SELECTION_FILE="${TRAIN_SELECTION_FILE:?set train selection JSONL}"
EVAL_SELECTION_FILE="${EVAL_SELECTION_FILE:?set FRESH selection JSONL}"
case "$SELECTOR" in
  bm25|llmlingua) ;;
  *) echo "unsupported SELECTOR=$SELECTOR" >&2; exit 2 ;;
esac
case "$CR" in
  8|16) ;;
  *) echo "unsupported CR=$CR" >&2; exit 2 ;;
esac
test -s "$TRAIN_SELECTION_FILE"
test -s "$EVAL_SELECTION_FILE"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${RUNS_ROOT}/sft_${SELECTOR}_cr${CR}_${STAMP}"
mkdir -p "$OUT" /home/aburlova/cvlm/runs/_logs
cd "$SNAP"
export PYTHONPATH="$SNAP/src"
export PYTHONUNBUFFERED=1

echo "=== SELECTED-SFT selector=$SELECTOR cr=$CR OUT=$OUT"
echo "=== train_selection=$TRAIN_SELECTION_FILE"
echo "=== eval_selection=$EVAL_SELECTION_FILE"
sha256sum src/train_sft.py src/eval_cvlm.py hse/protocol.env \
  hse/sft_selected_matched.sbatch "$TRAIN_SELECTION_FILE" "$EVAL_SELECTION_FILE" \
  > "$OUT/src_and_selection_hashes.txt"

python src/train_sft.py \
  --output_dir "$OUT" \
  --dataset_name sggetao/PwC \
  --dataset_split train \
  --model_name_or_path "$MODEL_NAME" \
  --max_prompt_len 512 \
  --max_answer_len 1024 \
  --truncate_cr "$CR" \
  --max_vision_len 1024 \
  --max_source_len 1024 \
  --text_encoder_name "$TEXT_ENCODER_NAME" \
  --doc_selection_file "$TRAIN_SELECTION_FILE" \
  --selection_policy file \
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
  --model_name_or_path "$MODEL_NAME" \
  --text_encoder_name "$TEXT_ENCODER_NAME" \
  --vision_encoder_name "$VISION_ENCODER_NAME" \
  --num_pool_latents 4 \
  --max_prompt_len 512 \
  --max_answer_len 1024 \
  --max_vision_len 1024 \
  --max_source_len 1024 \
  --max_samples 0 \
  --batch_size 8 \
  --seed 42 \
  --compression_rate 1 \
  --doc_truncation file \
  --doc_selection_file "$EVAL_SELECTION_FILE" \
  --output_json "$EOUT/${SELECTOR}_matched_reader_cr${CR}.json" \
  --dump_per_sample "$EOUT/${SELECTOR}_matched_reader_cr${CR}_dump.jsonl" \
  2>&1 | tee "$OUT/eval_fresh.log"

touch "$OUT/DONE"
echo "=== SELECTED_SFT_DONE selector=$SELECTOR cr=$CR OUT=$OUT"
