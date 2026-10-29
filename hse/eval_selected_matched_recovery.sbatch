#!/bin/bash
#SBATCH --job-name=selected_eval
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Eval-only recovery for a completed selected-reader training run.
# Required env: SELECTOR, CR, SFT_MODEL_PATH, EVAL_SELECTION_FILE.

set -euo pipefail

module purge
module load python/miniconda cuda/12.9.1
source activate /home/aburlova/conda_envs/cvlm

SNAP="${CVLM_SNAPSHOT_ROOT:-/home/aburlova/cvlm_reviewfix}"
source "$SNAP/hse/protocol.env"
unset EVAL_ALL_CHECKPOINTS EVAL_CR_SCHEDULE CR_SCHEDULE

SELECTOR="${SELECTOR:?set SELECTOR=bm25 or llmlingua}"
CR="${CR:?set CR=8 or CR=16}"
SFT_MODEL_PATH="${SFT_MODEL_PATH:?set the completed SFT model directory}"
EVAL_SELECTION_FILE="${EVAL_SELECTION_FILE:?set FRESH selection JSONL}"

case "$SELECTOR" in
  bm25|llmlingua) ;;
  *) echo "unsupported SELECTOR=$SELECTOR" >&2; exit 2 ;;
esac
case "$CR" in
  8|16) ;;
  *) echo "unsupported CR=$CR" >&2; exit 2 ;;
esac

test -f "$SFT_MODEL_PATH/TRAIN_DONE"
test -s "$EVAL_SELECTION_FILE"

EOUT="$SFT_MODEL_PATH/eval_fresh"
mkdir -p "$EOUT" /home/aburlova/cvlm/runs/_logs
cd "$SNAP"
export PYTHONPATH="$SNAP/src"
export PYTHONUNBUFFERED=1

echo "=== SELECTED-EVAL selector=$SELECTOR cr=$CR model=$SFT_MODEL_PATH"
echo "=== eval_selection=$EVAL_SELECTION_FILE"

python src/eval_cvlm.py \
  --mode sft \
  --sft_model_path "$SFT_MODEL_PATH" \
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
  2>&1 | tee "$SFT_MODEL_PATH/eval_fresh.log"

touch "$SFT_MODEL_PATH/DONE"
echo "=== SELECTED_EVAL_DONE selector=$SELECTOR cr=$CR model=$SFT_MODEL_PATH"
