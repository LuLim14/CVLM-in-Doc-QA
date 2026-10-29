#!/bin/bash
#SBATCH --job-name=review_smoke
#SBATCH --partition=test
#SBATCH --constraint=[type_e|type_f|type_h]
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Runtime smoke tests for the reviewer-response campaign. These runs are not
# scientific results. They exercise model/cache loading, the exact selection
# path, a real optimizer step at the production batch size, model saving, and
# (for reader arms) the production evaluator.

set -euo pipefail

module purge
module load python/miniconda cuda/12.9.1
source activate /home/aburlova/conda_envs/cvlm

SNAP="${CVLM_SNAPSHOT_ROOT:?set immutable snapshot path}"
MODE="${MODE:?set MODE=qwen|bm25|llmlingua}"
CR="${CR:-8}"
source "$SNAP/hse/protocol.env"
unset EVAL_ALL_CHECKPOINTS EVAL_CR_SCHEDULE CR_SCHEDULE

ROOT="/home/aburlova/cvlm/runs/review_smoke/${SLURM_JOB_ID}_${MODE}_cr${CR}"
mkdir -p "$ROOT" /home/aburlova/cvlm/runs/_logs
cd "$SNAP"
export PYTHONPATH="$SNAP/src"
export PYTHONUNBUFFERED=1
export TRACKIO_DISABLE=1

if [ "$MODE" = "llmlingua" ]; then
  python -c 'from importlib.metadata import version; assert version("llmlingua") == "0.2.2"'
  export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-/home/aburlova/tiktoken_cache}"
  python tools/llmlingua_select.py \
    --dataset_name sggetao/PwC \
    --dataset_split train \
    --decoder_tokenizer_name "$MODEL_NAME" \
    --encoder_tokenizer_name "$TEXT_ENCODER_NAME" \
    --max_prompt_len 512 \
    --max_answer_len 1024 \
    --max_source_len 1024 \
    --max_samples 64 \
    --budget_cr "$CR" \
    --max_vision_len 1024 \
    --llmlingua_model microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    --device cuda \
    --search_mode threshold \
    --search_iters 32 \
    --fidelity_check_docs 8 \
    --seed 42 \
    --output_jsonl "$ROOT/selection.jsonl" \
    --stats_json "$ROOT/stats.json"
  test -s "$ROOT/selection.jsonl"
  test -s "$ROOT/stats.json"
  touch "$ROOT/PASS"
  exit 0
fi

case "$MODE" in
  qwen)
    READER_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
    POLICY="first"
    ;;
  bm25)
    READER_MODEL="$MODEL_NAME"
    POLICY="query"
    ;;
  *)
    echo "unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$READER_MODEL', local_files_only=True); print('MODEL_CACHE_OK')"

MODEL_OUT="$ROOT/model"
python src/train_sft.py \
  --output_dir "$MODEL_OUT" \
  --dataset_name sggetao/PwC \
  --dataset_split train \
  --model_name_or_path "$READER_MODEL" \
  --max_prompt_len 512 \
  --max_answer_len 1024 \
  --max_samples 128 \
  --truncate_cr "$CR" \
  --selection_policy "$POLICY" \
  --max_vision_len 1024 \
  --max_source_len 1024 \
  --text_encoder_name "$TEXT_ENCODER_NAME" \
  --epochs 1 \
  --batch_size 64 \
  --gradient_accumulation_steps 1 \
  --lr 1e-4 \
  --lr_scheduler_type constant_with_warmup \
  --enable_warmup \
  --warmup_steps 100 \
  --grad_clip 1.0 \
  --gradient_checkpointing \
  --log_interval 1 \
  --save_interval_steps 0 \
  --seed 42

EVAL_EXTRA=()
if [ "$MODE" = "bm25" ]; then
  EVAL_EXTRA+=(--no_bf16)
fi
python src/eval_cvlm.py \
  --mode sft \
  --sft_model_path "$MODEL_OUT" \
  --dataset_name sggetao/PwC \
  --dataset_split 'test[2100:5100]' \
  --model_name_or_path "$READER_MODEL" \
  --text_encoder_name "$TEXT_ENCODER_NAME" \
  --vision_encoder_name "$VISION_ENCODER_NAME" \
  --num_pool_latents 4 \
  --max_prompt_len 512 \
  --max_answer_len 1024 \
  --max_vision_len 1024 \
  --max_source_len 1024 \
  --max_samples 32 \
  --batch_size 8 \
  --seed 42 \
  --compression_rate 1 \
  --doc_truncation "$POLICY" \
  --doc_budget_cr "$CR" \
  --output_json "$ROOT/eval.json" \
  --dump_per_sample "$ROOT/eval_dump.jsonl" \
  "${EVAL_EXTRA[@]}"

test -s "$ROOT/eval.json"
test -s "$ROOT/eval_dump.jsonl"
touch "$ROOT/PASS"
echo "REVIEW_SMOKE_PASS mode=$MODE cr=$CR root=$ROOT"
