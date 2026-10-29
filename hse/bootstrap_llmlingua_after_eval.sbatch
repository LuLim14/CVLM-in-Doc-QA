#!/bin/bash
#SBATCH --job-name=lingua_boot
#SBATCH --partition=test
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

set -euo pipefail

module purge
module load python/miniconda
source activate /home/aburlova/conda_envs/cvlm

REPORT=/home/aburlova/cvlm_reviewfix_lingua_nonmono_9be790d5/tools/llmlingua_report.py
RUNS=/home/aburlova/cvlm/runs
OUT="$RUNS/reviewer_response_20260927"
mkdir -p "$OUT" "$RUNS/_logs"

CR8_READER="$RUNS/sft_llmlingua_cr8_20260926_204740/eval_fresh/llmlingua_matched_reader_cr8_dump.jsonl"
CR16_READER="$RUNS/sft_llmlingua_cr16_20260926_204811/eval_fresh/llmlingua_matched_reader_cr16_dump.jsonl"
CR8_COMPRESSOR="$RUNS/fresh_heldout/compressor_cr8_dump.jsonl"
CR16_COMPRESSOR="$RUNS/fresh_heldout/compressor_cr16_dump.jsonl"

for path in "$REPORT" "$CR8_READER" "$CR16_READER" "$CR8_COMPRESSOR" "$CR16_COMPRESSOR"; do
  test -s "$path"
done

python "$REPORT" \
  --clusters_jsonl "$RUNS/llmlingua_baseline/llmlingua_cr8.jsonl" \
  --expect_rows 3000 \
  --expect_docs 205 \
  --cell llmlingua="$CR8_READER" \
  --cell compressor="$CR8_COMPRESSOR" \
  --contrast llmlingua:compressor \
  --n_boot 5000 \
  --seed 42 \
  --out_json "$OUT/bootstrap_llmlingua_matched_cr8.json"

python "$REPORT" \
  --clusters_jsonl "$RUNS/llmlingua_baseline/llmlingua_cr16.jsonl" \
  --expect_rows 3000 \
  --expect_docs 205 \
  --cell llmlingua="$CR16_READER" \
  --cell compressor="$CR16_COMPRESSOR" \
  --contrast llmlingua:compressor \
  --n_boot 5000 \
  --seed 42 \
  --out_json "$OUT/bootstrap_llmlingua_matched_cr16.json"

echo "=== LLMLINGUA_MATCHED_BOOTSTRAP_DONE ==="
