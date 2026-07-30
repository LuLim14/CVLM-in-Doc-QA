#!/bin/bash
#SBATCH --job-name=cvlm_paddiag
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# РЕШАЮЩАЯ ДИАГНОСТИКА: объясняется ли depth penalty контаминацией паддингом?
# Гоняется на УЖЕ ОБУЧЕННЫХ чекпоинтах боевых веток, переобучение не нужно.
# Флаги ветки повторяют обучение (mask_vit_padding=false = legacy-поведение).
set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs
module purge; module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm
source /home/aburlova/cvlm_rd4/hse/protocol.env
export CVLM_ROOT=/home/aburlova/cvlm_rd4
cd "$CVLM_ROOT"
export PYTHONPATH="$CVLM_ROOT/src" PYTHONUNBUFFERED=1
OUT=/home/aburlova/cvlm/runs/padding_diag; mkdir -p "$OUT"
# Провенанс: версия кода + хеши исходников рядом с результатами (immutability audit).
cp "$CVLM_ROOT/VERSION" "$OUT/VERSION" 2>/dev/null || true
sha256sum "$CVLM_ROOT"/src/*.py > "$OUT/src_hashes.txt"
for B in novit randomvit vit; do
  RD=/home/aburlova/cvlm/runs/abl_${B}_20260729_171933
  CK=$(ls -1 $RD/model_step_*.safetensors 2>/dev/null | sed -E "s|.*model_step_([0-9]+)\.safetensors|\1 &|" | sort -n | tail -1 | awk '{print $2}')
  [ -z "$CK" ] && continue
  VI=pretrained; NV=false
  case $B in randomvit) VI=random;; novit) NV=true;; esac
  for CR in 1 16; do
    J="$OUT/${B}_cr${CR}.json"; [ -s "$J" ] && continue
    echo "=== $B cr=$CR"
    python tools/measure_padding_contamination.py \
      --checkpoint_path "$CK" --compression_rate "$CR" \
      --model_name_or_path "$MODEL_NAME" --text_encoder_name "$TEXT_ENCODER_NAME" \
      --vision_encoder_name "$VISION_ENCODER_NAME" \
      --num_pool_latents 4 --max_vision_len 1024 --max_source_len 1024 \
      --max_prompt_len 512 --max_answer_len 1024 --max_samples 300 \
      --vit_init $VI --no_vit $NV --mask_vit_padding false \
      --output_json "$J" || echo "FAILED $B cr=$CR"
  done
done
echo PADDING_DIAG_DONE
