#!/bin/bash
#SBATCH --job-name=sft_matched
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# MATCHED-TRAINING truncated-SFT — ответ на критику ревьюера («единственный
# серьёзный противник, возможно, недоучен»): тот же ОПТИМИЗАЦИОННЫЙ протокол,
# что у компрессора (hse/protocol.env), для truncated-SFT бейзлайна.
#
#   совпадает с компрессором:  3 эпохи; eff.batch 64 (bs=64 x accum=1);
#     lr=1e-4; linear warmup 100 шагов -> КОНСТАНТА (эквивалент
#     DISABLE_LR_SCHEDULE=1); bf16; grad_clip=1.0; AdamW wd=0,
#     betas=(0.9,0.999), eps=1e-8; gradient checkpointing; seed 42;
#     те же фильтры данных (prompt<=512, answer<=1024, source_enc<=1024,
#     kept 241564) => ~11325 optimizer-шагов (3*ceil(241564/64)).
#   отличия от СТАРОГО sft_trunc* (runs 20260730_072938, jobs 4206872/3):
#     epochs 2->3, lr 1e-5->1e-4, cosine-decay-to-0 без warmup ->
#     constant после 100-шагового warmup. Батч (64), данные, усечение
#     first-N до min(ceil(L_enc/cr), 1024) и маскировка НЕ менялись.
#   НЕ переносится от компрессора (осознанно): cr-куррикулум 1->2->4->8->16 —
#     truncated-SFT по определению обучается на фиксированном бюджете cr.
#
# train_sft.py жёстко зашивает lr_scheduler_type="cosine" => шим-обёртка
# (heredoc ниже) монки-патчит TrainingArguments.__init__ на
# constant_with_warmup ДО запуска train_sft.py; код снапшота не меняется.
#
# ВНИМАНИЕ (задокументированный риск): lr=1e-4 для full-FT 1.7B-декодера --
# агрессивно; это ТРЕБОВАНИЕ дизайна matched-протокола. Лосс пишется в
# metrics.csv каждые 10 шагов -- дивергенция видна сразу. При дивергенции
# джобу НЕ снимать: это тоже результат.
#
# Запуск:  sbatch --job-name=sft_matched_cr16 --export=ALL,CR=16 sft_matched.sbatch
#          sbatch --job-name=sft_matched_cr8  --export=ALL,CR=8  sft_matched.sbatch

set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs
module purge; module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm

SNAP=/home/aburlova/cvlm_rd11
source "$SNAP/hse/protocol.env"
export CVLM_ROOT="$SNAP"
# env-ловушки eval_cvlm.py: argparse-default читает EVAL_ALL_CHECKPOINTS /
# EVAL_CR_SCHEDULE прямо из env, а --all_checkpoints НЕСОВМЕСТИМ с --mode sft
# (ValueError). Глушим до любого python.
unset EVAL_ALL_CHECKPOINTS EVAL_CR_SCHEDULE CR_SCHEDULE

CR="${CR:?set CR=16|8 via sbatch --export=ALL,CR=...}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${RUNS_ROOT}/sft_matched_cr${CR}_${STAMP}"
mkdir -p "$OUT"
cp "$SNAP/VERSION" "$OUT/VERSION"
sha256sum "$SNAP"/src/train_sft.py "$SNAP"/src/eval_cvlm.py "$SNAP"/src/train_logging.py \
  "$SNAP"/hse/protocol.env > "$OUT/src_hashes.txt"
echo "=== SFT-MATCHED CR=$CR OUT=$OUT VERSION=$(cat "$SNAP/VERSION")"
sed 's/^/    sha256 /' "$OUT/src_hashes.txt"
echo "=== node $(hostname), gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

cd "$SNAP"
export PYTHONPATH="$SNAP/src" PYTHONUNBUFFERED=1

# --- шим: constant-after-warmup вместо зашитого cosine ---------------------
cat > "$OUT/train_sft_constlr.py" <<'PYEOF'
"""Matched-training шим: запускает train_sft.py снапшота как __main__, но
принудительно lr_scheduler_type='constant_with_warmup' (линейный warmup до
init_lr, дальше константа) -- семантика warmup+DISABLE_LR_SCHEDULE=1
компрессора (train_cvlm.py). Снапшот кода не модифицируется."""
import os
import runpy

import transformers

_orig_init = transformers.TrainingArguments.__init__


def _patched_init(self, *args, **kwargs):
    was = kwargs.get("lr_scheduler_type", "<unset>")
    kwargs["lr_scheduler_type"] = "constant_with_warmup"
    print(f"[shim] TrainingArguments: lr_scheduler_type {was!r} -> "
          f"'constant_with_warmup' (warmup_steps={kwargs.get('warmup_steps')})",
          flush=True)
    return _orig_init(self, *args, **kwargs)


transformers.TrainingArguments.__init__ = _patched_init

_root = os.environ.get("CVLM_ROOT", "/home/aburlova/cvlm_rd11")
runpy.run_path(os.path.join(_root, "src", "train_sft.py"), run_name="__main__")
PYEOF

# --- обучение ---------------------------------------------------------------
python "$OUT/train_sft_constlr.py" \
  --output_dir "$OUT" \
  --dataset_name sggetao/PwC \
  --dataset_split train \
  --model_name_or_path "$MODEL_NAME" \
  --max_prompt_len "$MAX_PROMPT_LEN" \
  --max_answer_len "$MAX_ANSWER_LEN" \
  --truncate_cr "$CR" \
  --max_vision_len "$MAX_VISION_LEN" \
  --max_source_len "$MAX_SOURCE_LEN" \
  --text_encoder_name "$TEXT_ENCODER_NAME" \
  --epochs 3 \
  --batch_size 64 \
  --gradient_accumulation_steps 1 \
  --lr 1e-4 \
  --enable_warmup --warmup_steps 100 \
  --grad_clip 1.0 \
  --gradient_checkpointing \
  --log_interval 10 \
  --save_interval_steps 1000 \
  --seed "$SEED" \
  2>&1 | tee "$OUT/pipeline.log"

# --- встроенный eval: teacher-forced mode=sft на FRESH test[2100:5100] ------
# Инвокация байт-в-байт по freshheldout.sbatch (фаза 2, sfttrunc_cr*), только
# модель -- свежеобученная. Дамп per-sample нужен для парного bootstrap.
EOUT="$OUT/eval_fresh"; mkdir -p "$EOUT"
python src/eval_cvlm.py --mode sft --sft_model_path "$OUT" \
  --dataset_name sggetao/PwC --dataset_split "test[2100:5100]" \
  --model_name_or_path "$MODEL_NAME" \
  --text_encoder_name "$TEXT_ENCODER_NAME" \
  --vision_encoder_name "$VISION_ENCODER_NAME" \
  --num_pool_latents 4 --max_prompt_len 512 --max_answer_len 1024 \
  --max_vision_len 1024 --max_source_len 1024 --max_samples 0 \
  --batch_size 8 --seed 42 \
  --compression_rate 1 --doc_truncation first --doc_budget_cr "$CR" \
  --output_json "$EOUT/sftmatched_cr${CR}.json" \
  --dump_per_sample "$EOUT/sftmatched_cr${CR}_dump.jsonl" \
  2>&1 | tee "$OUT/eval_fresh.log"

echo "=== SFT_MATCHED_DONE CR=$CR OUT=$OUT"
