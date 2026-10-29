#!/usr/bin/env bash
# Предзагрузка HF-артефактов на ЛОГИН-ноде cHARISMa (компьют-ноды без интернета).
# Запуск: bash prefetch.sh 2>&1 | tee prefetch.log
set -euo pipefail

module purge
module load Python/Miniconda
source activate /home/aburlova/conda_envs/cvlm

export HF_HOME=/home/aburlova/hf_cache
export HF_HUB_CACHE=/home/aburlova/hf_cache/hub
export HUGGINGFACE_HUB_CACHE=/home/aburlova/hf_cache/hub
export HF_HUB_OFFLINE=0

for repo in \
  HuggingFaceTB/SmolLM2-1.7B-Instruct \
  HuggingFaceTB/SmolLM-135M-Instruct \
  answerdotai/ModernBERT-base \
  google/vit-base-patch16-224; do
  echo "=== model: ${repo}"
  huggingface-cli download "${repo}" >/dev/null
done

echo "=== dataset: sggetao/PwC"
huggingface-cli download sggetao/PwC --repo-type dataset >/dev/null

# Прогрев Arrow-кэша + КРИТИЧЕСКАЯ проверка наличия сплита test (eval падает без него).
python - <<'PY'
from datasets import load_dataset
tr = load_dataset("sggetao/PwC", split="train")
print("train rows:", len(tr), "cols:", tr.column_names)
try:
    te = load_dataset("sggetao/PwC", split="test")
    print("test rows:", len(te))
except Exception as e:
    raise SystemExit(f"FATAL: no 'test' split in sggetao/PwC: {e}\n"
                     "-> нужно решить, чем евалить (например держаный хвост train) ДО запуска веток")
PY

echo "=== offline sanity (как увидят компьют-ноды)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python - <<'PY'
from transformers import AutoTokenizer, AutoConfig
for m in ["HuggingFaceTB/SmolLM2-1.7B-Instruct", "answerdotai/ModernBERT-base", "google/vit-base-patch16-224"]:
    AutoConfig.from_pretrained(m); print("config ok:", m)
AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct", use_fast=False)
print("slow tokenizer ok (vocab/merges present)")
from datasets import load_dataset
load_dataset("sggetao/PwC", split="train"); print("offline dataset ok")
PY
echo "PREFETCH DONE"
