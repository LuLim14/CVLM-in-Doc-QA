#!/usr/bin/env bash
# Run ModernBERT embedding preprocess, then CVLM training on the saved .npy.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# --- defaults (override with env or edit below) ---
EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-${ROOT}/data/processed/embeddings.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/cvlm_$(date +%Y%m%d_%H%M%S)}"
DATASET_NAME="${DATASET_NAME:-sggetao/PwC}"
EMBEDDER_MODEL="${EMBEDDER_MODEL:-answerdotai/ModernBERT-base}"
EMBED_MAX_LENGTH="${EMBED_MAX_LENGTH:-2048}"
EMBED_MAX_SAMPLES="${EMBED_MAX_SAMPLES:-0}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-32}"

# torchrun when NPROC>1, else single-process python
NPROC="${NPROC:-1}"

mkdir -p "$(dirname "$EMBEDDINGS_PATH")"
mkdir -p "$OUTPUT_DIR"

echo "======== Step 1/2: Build embeddings -> ${EMBEDDINGS_PATH} ========"
python "${ROOT}/build_dataset/embed_dataset.py" \
  --embeddings_path "$EMBEDDINGS_PATH" \
  --dataset_name "$DATASET_NAME" \
  --embedder_model "$EMBEDDER_MODEL" \
  --max_length "$EMBED_MAX_LENGTH" \
  --max_samples "$EMBED_MAX_SAMPLES" \
  --batch_size "$EMBED_BATCH_SIZE"

echo "======== Step 2/2: Train CVLM -> ${OUTPUT_DIR} ========"
TRAIN_ARGS=(
  "${ROOT}/src/train_cvlm.py"
  --output_dir "$OUTPUT_DIR"
  --embeddings_path "$EMBEDDINGS_PATH"
  --dataset_name "$DATASET_NAME"
)

if [[ "${NPROC}" -gt 1 ]]; then
  exec torchrun --nproc_per_node="${NPROC}" "${TRAIN_ARGS[@]}" "$@"
else
  exec python "${TRAIN_ARGS[@]}" "$@"
fi
