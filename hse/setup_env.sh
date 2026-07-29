#!/usr/bin/env bash
# Создание conda-env `cvlm` на ЛОГИН-ноде cHARISMa (интернет есть только там).
# Вариант A из плана: torch 2.5.1+cu124 + liger-kernel 0.8.x — авторский сетап.
# Перед запуском: проверить драйвер (nvidia-smi на компьют-ноде >= 550 для cu124;
# если старее — заменить cu124 на cu121 ниже).
# Запуск: bash setup_env.sh 2>&1 | tee setup_env.log
set -euo pipefail

module purge
module load Python/Miniconda

ENV_DIR=/home/aburlova/conda_envs/cvlm

if [ -d "${ENV_DIR}" ]; then
  echo "ENV ${ENV_DIR} already exists — remove it first if you want a rebuild."
else
  conda create -y -p "${ENV_DIR}" python=3.11
fi

source activate "${ENV_DIR}"

pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# ЛОГИН-НОДА: glibc 2.17 — свежие numpy/pillow/pyarrow шипят только manylinux_2_28
# колёса и падают в сборку из исходников (GCC<9.3). --prefer-binary заставляет pip
# взять старейшую версию С СОВМЕСТИМЫМ КОЛЕСОМ вместо сборки новейшего sdist.
pip install --no-cache-dir --prefer-binary "numpy<2"
# Пины эпохи manylinux2014 (у более новых версий колёса только manylinux_2_28,
# а на логин-ноде glibc 2.17 -> pip уходит в сборку из исходников и падает):
pip install --no-cache-dir --prefer-binary \
  "pyarrow==14.0.2" \
  "pillow==10.4.0" \
  "matplotlib==3.8.4"
pip install --no-cache-dir --prefer-binary \
  "transformers==4.57.6" \
  accelerate \
  "liger-kernel>=0.8.0,<0.9" \
  "datasets==2.19.2" \
  safetensors \
  sacrebleu \
  rouge-score \
  tqdm \
  pytest

python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda build:", torch.version.cuda)
print("transformers", transformers.__version__)
import liger_kernel; print("liger ok")
import datasets, sacrebleu, rouge_score, matplotlib
print("deps ok")
PY
echo "ENV READY: source activate ${ENV_DIR}"
