#!/bin/bash
#SBATCH --job-name=bm25_sft
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Distribution-matched reader for the exact BM25-style selector used by the
# headline query-aware baseline (original Slurm job 4229309).

set -euo pipefail

module purge
module load python/miniconda cuda/12.9.1
source activate /home/aburlova/conda_envs/cvlm

SNAP="${CVLM_SNAPSHOT_ROOT:?set immutable snapshot path}"
source "$SNAP/hse/protocol.env"
unset EVAL_ALL_CHECKPOINTS EVAL_CR_SCHEDULE CR_SCHEDULE

CR="${CR:?set CR=8 or CR=16}"
case "$CR" in
  8|16) ;;
  *) echo "unsupported CR=$CR" >&2; exit 2 ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${RUNS_ROOT}/sft_bm25_cr${CR}_${STAMP}"
mkdir -p "$OUT" /home/aburlova/cvlm/runs/_logs
cd "$SNAP"
export PYTHONPATH="$SNAP/src"
export PYTHONUNBUFFERED=1

echo "=== BM25-MATCHED-SFT CR=$CR SNAP=$SNAP OUT=$OUT"
echo "selector_source_job=4229309" > "$OUT/PROVENANCE"
echo "selector=k1=1.5,b=0.75,sentence_regex,greedy_cap,document_order" >> "$OUT/PROVENANCE"
sha256sum src/train_sft.py src/cvlm_dataset.py src/eval_cvlm.py \
  hse/protocol.env hse/sft_bm25_matched.sbatch > "$OUT/src_hashes.txt"

python src/train_sft.py \
  --output_dir "$OUT" \
  --dataset_name sggetao/PwC \
  --dataset_split train \
  --model_name_or_path "$MODEL_NAME" \
  --max_prompt_len 512 \
  --max_answer_len 1024 \
  --truncate_cr "$CR" \
  --selection_policy query \
  --max_vision_len 1024 \
  --max_source_len 1024 \
  --text_encoder_name "$TEXT_ENCODER_NAME" \
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
  --doc_truncation query \
  --doc_budget_cr "$CR" \
  --no_bf16 \
  --output_json "$EOUT/bm25_matched_reader_cr${CR}.json" \
  --dump_per_sample "$EOUT/bm25_matched_reader_cr${CR}_dump.jsonl" \
  2>&1 | tee "$OUT/eval_fresh.log"

touch "$OUT/DONE"
echo "=== BM25_MATCHED_SFT_DONE CR=$CR OUT=$OUT"
