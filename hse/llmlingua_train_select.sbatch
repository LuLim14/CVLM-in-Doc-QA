#!/bin/bash
#SBATCH --job-name=lingua_train_select
#SBATCH --partition=rocky
#SBATCH --constraint=[type_a|type_b|type_c]
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Generate and audit LLMLingua-2 selections for the full PwC training split.
# The two downstream SFT jobs depend on successful completion of this job.

set -euo pipefail

module purge
module load python/miniconda cuda/12.9.1
source activate /home/aburlova/conda_envs/cvlm

SNAP="${CVLM_SNAPSHOT_ROOT:?set immutable snapshot path}"
source "$SNAP/hse/protocol.env"
unset EVAL_ALL_CHECKPOINTS EVAL_CR_SCHEDULE CR_SCHEDULE

LLMLINGUA_PKG_VERSION="$(python -c 'from importlib.metadata import version; print(version("llmlingua"))')"
test "$LLMLINGUA_PKG_VERSION" = "0.2.2"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-/home/aburlova/tiktoken_cache}"
test -n "$(ls -A "$TIKTOKEN_CACHE_DIR")"

OUT="${LLMLINGUA_TRAIN_OUT:-/home/aburlova/cvlm/runs/llmlingua_train_select}"
mkdir -p "$OUT" /home/aburlova/cvlm/runs/_logs
cd "$SNAP"
export PYTHONPATH="$SNAP/src"
export PYTHONUNBUFFERED=1

sha256sum tools/llmlingua_select.py tools/audit_selection_jsonl.py \
  hse/llmlingua_train_select.sbatch > "$OUT/src_hashes.txt"

if [ -n "${CR_ONLY:-}" ]; then
  case "$CR_ONLY" in
    8|16) CR_LIST="$CR_ONLY" ;;
    *) echo "unsupported CR_ONLY=$CR_ONLY" >&2; exit 2 ;;
  esac
else
  CR_LIST="8 16"
fi

for CR in $CR_LIST; do
  SEL="$OUT/llmlingua_train_cr${CR}.jsonl"
  STATS="$OUT/llmlingua_train_cr${CR}_stats.json"
  AUDIT="$OUT/llmlingua_train_cr${CR}_audit.json"
  if [ ! -s "$SEL" ] || [ ! -s "$STATS" ]; then
    python tools/llmlingua_select.py \
      --dataset_name sggetao/PwC \
      --dataset_split train \
      --decoder_tokenizer_name "$MODEL_NAME" \
      --encoder_tokenizer_name "$TEXT_ENCODER_NAME" \
      --max_prompt_len 512 \
      --max_answer_len 1024 \
      --max_source_len 1024 \
      --max_samples 0 \
      --budget_cr "$CR" \
      --max_vision_len 1024 \
      --llmlingua_model microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
      --device cuda \
      --search_mode threshold \
      --search_iters 32 \
      --fidelity_check_docs 25 \
      --seed 42 \
      --output_jsonl "$SEL" \
      --stats_json "$STATS"
  fi
  python tools/audit_selection_jsonl.py \
    --jsonl "$SEL" \
    --budget_cr "$CR" \
    --max_vision_len 1024 \
    --dataset_name sggetao/PwC \
    --dataset_split train \
    --decoder_tokenizer_name "$MODEL_NAME" \
    --encoder_tokenizer_name "$TEXT_ENCODER_NAME" \
    --max_prompt_len 512 \
    --max_answer_len 1024 \
    --max_source_len 1024 \
    --max_samples 0 \
    --expect_rows 241564 \
    --out_json "$AUDIT"
done

if [ -n "${CR_ONLY:-}" ]; then
  touch "$OUT/DONE_cr${CR_ONLY}"
  echo "=== LLMLINGUA_TRAIN_SELECTION_DONE CR=$CR_ONLY OUT=$OUT"
else
  touch "$OUT/DONE"
  echo "=== LLMLINGUA_TRAIN_SELECTION_DONE OUT=$OUT"
fi
