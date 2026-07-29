#!/usr/bin/env bash
# Боевой запуск всех веток (после успешного смоука!): партиция cpu-e-quick
# (A100, MaxTime 24h; rocky для аккаунта отдаёт только V100 — проверено 29.07.2026).
# Каждой ветке — фиксированный OUTPUT_DIR + chained-джоба продолжения по
# afterany: если первичная упёрлась в 24h/упала — вторая резюмит с чекпоинта;
# если добежала (TRAIN_DONE) — вторая мгновенно выходит.
# Запуск: bash launch_all.sh            (4 абляции + 2 truncated-SFT)
#         WITH_SFT_FULL=1 bash launch_all.sh   (+ full-doc SFT ceiling)
set -euo pipefail
cd "$(dirname "$0")"
source ./protocol.env

STAMP="$(date +%Y%m%d_%H%M%S)"
for BR in vit randomvit novit frozenvit; do
  OUT="${RUNS_ROOT}/abl_${BR}_${STAMP}"
  JID=$(sbatch --parsable --job-name="abl_${BR}" \
        --export=ALL,BRANCH="${BR}",OUTPUT_DIR="${OUT}" train_branch.sbatch)
  sbatch --job-name="abl_${BR}_c" --dependency="afterany:${JID}" \
        --export=ALL,BRANCH="${BR}",OUTPUT_DIR="${OUT}" train_branch.sbatch >/dev/null
  echo "abl_${BR}: job ${JID} (+chained resume) -> ${OUT}"
done

sbatch --job-name=sft_trunc8  --export=ALL,TRUNC_CR=8  sft_branch.sbatch
sbatch --job-name=sft_trunc16 --export=ALL,TRUNC_CR=16 sft_branch.sbatch

if [ "${WITH_SFT_FULL:-0}" = "1" ]; then
  sbatch --job-name=sft_full --export=ALL,TRUNC_CR=0 sft_branch.sbatch
fi

squeue -u aburlova -o '%i %j %P %T %M %R'
