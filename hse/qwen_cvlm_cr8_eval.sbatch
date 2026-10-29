#!/bin/bash
#SBATCH --job-name=qwen_cvlm_cr8
#SBATCH --partition=test
#SBATCH --constraint=[type_e|type_f|type_h]
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# Missing cr=8 cell for the existing Qwen Pool+Project checkpoint. Keep the
# exact code snapshot and numerical eval mode used for qwen_fresh cr=16.
set -euo pipefail

module purge
module load python/miniconda cuda/12.9.1
source activate /home/aburlova/conda_envs/cvlm

SNAP=/home/aburlova/cvlm_qwen
CK=/home/aburlova/cvlm/runs/qwen05b_novit_20260801_152219/model_step_11326.safetensors
OUT=/home/aburlova/cvlm/runs/qwen_fresh
J="$OUT/novit_cvlm_cr8.json"
D="$OUT/novit_cvlm_cr8_dump.jsonl"

unset EVAL_ALL_CHECKPOINTS EVAL_CR_SCHEDULE CR_SCHEDULE
cd "$SNAP"
export PYTHONPATH="$SNAP/src"
export PYTHONUNBUFFERED=1
export DISABLE_LIGER=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export TRITON_CACHE_DIR="/tmp/triton_$SLURM_JOB_ID"
export CVLM_EVAL_FP16=1
export HF_HOME=/home/aburlova/hf_cache
export HF_HUB_CACHE=/home/aburlova/hf_cache/hub
export HUGGINGFACE_HUB_CACHE=/home/aburlova/hf_cache/hub
export HF_HUB_OFFLINE=1

test -s "$CK"
mkdir -p "$OUT" /home/aburlova/cvlm/runs/_logs
test ! -e "$J"
test ! -e "$D"

python src/eval_cvlm.py \
  --mode cvlm \
  --checkpoint_path "$CK" \
  --compression_rate 8 \
  --dataset_name sggetao/PwC \
  --dataset_split 'test[2100:5100]' \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
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
  --global_step 11326 \
  --vit_init pretrained \
  --freeze_vit 0 \
  --no_vit 1 \
  --output_json "$J" \
  --dump_per_sample "$D"

test -s "$J"
test "$(wc -l < "$D")" -eq 3000
echo "QWEN_CVLM_CR8_DONE json=$J dump=$D"
