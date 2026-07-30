#!/bin/bash
#SBATCH --job-name=cvlm_mask
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# РЕШАЮЩИЙ ЭКСПЕРИМЕНТ ПРО КОНФАУНДЕР: снимает ли pad-маскирование штраф за глубину?
#   ARM=novit    : pooling-only (эталон; ViT-путь не задействован)
#   ARM=unmasked : random-init 12-слойный ViT БЕЗ маски (воспроизводит боевую ветку)
#   ARM=masked   : тот же ViT С key-padding маской
# Единственная разница unmasked/masked — маска. Всё остальное идентично
# (seed, батчи, shared_init_seed, куррикулум).
set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs
module purge; module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm
source /home/aburlova/cvlm_rd4/hse/protocol.env
export CVLM_ROOT=/home/aburlova/cvlm_rd4
ARM="${ARM:?set ARM=novit|unmasked|masked}"
export VIT_INIT=random FREEZE_VIT=0 NO_VIT=0 MASK_VIT_PADDING=0 MIXER_DEPTH_MAP=""
case "$ARM" in
  novit)    export NO_VIT=1 ;;
  unmasked) export MASK_VIT_PADDING=0 ;;
  masked)   export MASK_VIT_PADDING=1 ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac
export RUN_TAG="mask_${ARM}"
export OUTPUT_DIR="${RUNS_ROOT}/mask_${ARM}_${MC_STAMP:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
cp "$CVLM_ROOT/VERSION" "$OUTPUT_DIR/VERSION" 2>/dev/null || true
sha256sum "$CVLM_ROOT"/src/*.py > "$OUTPUT_DIR/src_hashes.txt"
# Один финальный eval вместо перебора всех чекпоинтов (экономия walltime).
export EVAL_ALL_CHECKPOINTS=0
echo "=== ARM=$ARM MASK_VIT_PADDING=$MASK_VIT_PADDING NO_VIT=$NO_VIT OUT=$OUTPUT_DIR"
cd "$CVLM_ROOT"; bash scripts/run_full_pipeline.sh
ls -1 "$OUTPUT_DIR"/trainer_step_*.pt 2>/dev/null | sort -t_ -k3 -n | head -n -1 | xargs -r rm -f
echo "=== MASKCMP DONE $ARM"
