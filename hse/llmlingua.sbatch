#!/bin/bash
#SBATCH --job-name=cvlm_lingua
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aburlova/cvlm/runs/_logs/%x_%j.out
#SBATCH --error=/home/aburlova/cvlm/runs/_logs/%x_%j.err

# ============================================================================
# LLMLingua-2 — ВНЕШНИЙ ОПУБЛИКОВАННЫЙ БЕЙЗЛАЙН СЖАТИЯ ПРОМПТА
# ============================================================================
# Зачем. Все текстовые бейзлайны RQ3 (first/last/random/oracle, skim-then-read)
# — наши собственные. Рецензенту нужен ОПУБЛИКОВАННЫЙ конкурент в текстовом
# пространстве: LLMLingua-2 (Pan et al., ACL Findings 2024),
# microsoft/llmlingua-2-xlm-roberta-large-meetingbank.
#
# СТРОГО РАВНЫЙ БЮДЖЕТ. Родной параметр LLMLingua — rate в ЕЁ токенизации
# (XLM-R / cl100k), а не в слотах декодера. Поэтому tools/llmlingua_select.py
# калибрует rate под число латентов CVLM
#     V_i = max(min(ceil(min(L_ModernBERT_i, MAX_SOURCE_LEN)/CR), MAX_VISION_LEN), 1),
# измеренное в токенах ТОКЕНИЗАТОРА ДЕКОДЕРА (SmolLM2), и жёстко обрезает
# результат до <= V_i. Формула побитово та же, что в
# CvlmTrainDataset._truncated_doc_ids (--doc_budget_cr).
#
# Сжимается ТОЛЬКО документ (поле input). Вопрос (поле prompt) в компрессор не
# передаётся вообще: CvlmTrainDataset сам приклеивает "\n\n"+вопрос к
# selected_text, как и для всех остальных стратегий doc_truncation.
#
# ЧЕТЫРЕ ФАЗЫ:
#   1) СЕЛЕКЦИЯ (GPU): LLMLingua-2 -> JSONL {"idx","selected_text",...} в
#      формате --doc_selection_file (контракт tools/score_chunks.py; код
#      eval_cvlm.py и cvlm_dataset.py НЕ тронут).
#   2) АУДИТ (CPU): tools/audit_selection_jsonl.py. Загрузчик датасета
#      проверяет ТОЛЬКО число строк и диапазон idx — переставленный или
#      устаревший файл он не поймает, и бюджет он не перемеряет. Аудитор
#      сверяет sha256 документа И вопроса построчно с датасетом, пересчитывает
#      V_i, перемеряет длину selected_text токенизатором декодера и проверяет
#      инвариантность отбора к вопросу (в PwC один документ идёт с ~15
#      вопросами; отбор по документу обязан быть побитово одинаковым).
#      Любая фатальная находка валит джобу ДО оценок.
#   3) ЧЕТЫРЕ ОЦЕНКИ = 2 cr (8,16) x 2 декодера:
#        (а) diag_pretrained_decoder — предобученный SmolLM2-1.7B-Instruct,
#            КАК ОПУБЛИКОВАН: --checkpoint_path НЕ передаётся, поэтому веса
#            декодера НЕ из CVLM-чекпоинта (в CVLM декодер обучаемый — 97.94%
#            обучаемых параметров — и клетка с чекпоинтом называлась бы
#            "CVLM-trained decoder"; здесь это НЕ так, см. LABELS).
#            Вспомогательная клетка: текстовый вход для декодера,
#            не видевшего такого формата.
#        (б) primary_sfttrunc_decoder — truncated-SFT декодер того же cr
#            (runs/sft_trunc${CR}_*). ОСНОВНОЙ результат LLMLingua: декодер
#            обучен ровно на "усечённый документ + вопрос", поэтому бейзлайн
#            не штрафуется за непривычный формат входа.
#   4) ОТЧЁТ (CPU): tools/llmlingua_report.py — первичная величина NLL
#      (нат/answer-токен, агрегировано sum/sum), PPL = exp(АГРЕГИРОВАННОГО
#      NLL), парный bootstrap кластерами по ДОКУМЕНТАМ (строки не независимы:
#      ~15 вопросов на документ), включая парные контрасты с уже посчитанными
#      клетками того же свежего сплита из runs/fresh_heldout.
#
# Сплит — свежий held-out test[2100:5100] (3000 строк / 205 документов),
# max_samples=0, batch_size 8, seed 42, пер-сэмпловые дампы обязательны.
#
# ОФЛАЙН. Компьют-ноды без интернета. Модель LLMLingua-2 лежит в HF_HOME,
# HF_HUB_OFFLINE=1. ОТДЕЛЬНАЯ МИНА: llmlingua.PromptCompressor.__init__
# безусловно делает tiktoken.encoding_for_model("gpt-3.5-turbo"), а tiktoken
# качает BPE-файл по сети. Кэш заранее прогрет на логин-ноде в
# TIKTOKEN_CACHE_DIR; здесь он экспортируется и проверяется на существование.
#
# ОШИБКИ НЕ ГЛУШИМ (set -euo pipefail, никаких "|| echo"): любая упавшая
# клетка валит джобу, чтобы не получить ложный DONE с дырой в таблице.
#
# Запуск: sbatch hse/llmlingua.sbatch
# ============================================================================

set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs

module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm

SNAP="${CVLM_SNAPSHOT_ROOT:-/home/aburlova/cvlm_rd9}"
source "${SNAP}/hse/protocol.env"
export CVLM_ROOT="${SNAP}"
# protocol.env — про ОБУЧЕНИЕ: расписание cr и обход всех чекпоинтов здесь
# только навредят (eval_cvlm.py читает их из окружения как дефолты).
unset EVAL_ALL_CHECKPOINTS EVAL_CR_SCHEDULE CR_SCHEDULE

test -s "${SNAP}/VERSION" || { echo "ERROR: missing ${SNAP}/VERSION"; exit 1; }
test -s "${SNAP}/tools/llmlingua_select.py" || { echo "ERROR: sync tools/llmlingua_select.py first"; exit 1; }
test -s "${SNAP}/tools/audit_selection_jsonl.py" || { echo "ERROR: sync tools/audit_selection_jsonl.py first"; exit 1; }
test -s "${SNAP}/tools/llmlingua_report.py" || { echo "ERROR: sync tools/llmlingua_report.py first"; exit 1; }
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-/home/aburlova/tiktoken_cache}"
test -d "${TIKTOKEN_CACHE_DIR}" || { echo "ERROR: no tiktoken cache at ${TIKTOKEN_CACHE_DIR}; prewarm on the login node"; exit 1; }
test -n "$(ls -A "${TIKTOKEN_CACHE_DIR}")" || { echo "ERROR: empty tiktoken cache ${TIKTOKEN_CACHE_DIR}"; exit 1; }

OUT="${LLMLINGUA_OUT:-/home/aburlova/cvlm/runs/llmlingua_baseline}"
REF="${LLMLINGUA_REF:-/home/aburlova/cvlm/runs/fresh_heldout}"
mkdir -p "${OUT}"
cd "${SNAP}"
export PYTHONPATH="${SNAP}/src" PYTHONUNBUFFERED=1

cp "${SNAP}/VERSION" "${OUT}/CODE_VERSION"
sha256sum \
  tools/llmlingua_select.py tools/audit_selection_jsonl.py tools/llmlingua_report.py \
  src/eval_cvlm.py src/cvlm_dataset.py src/modeling.py \
  hse/llmlingua.sbatch hse/protocol.env \
  > "${OUT}/CODE_SHA256"

SPLIT="test[2100:5100]"
LINGUA_MODEL=microsoft/llmlingua-2-xlm-roberta-large-meetingbank
{
  echo "node=$(hostname)"
  echo "job=${SLURM_JOB_ID:-none}"
  echo "code_version=$(cat "${SNAP}/VERSION")"
  echo "split=${SPLIT}"
  echo "llmlingua_model=${LINGUA_MODEL}"
  echo "llmlingua_pkg_version=$(python -c 'import llmlingua; print(llmlingua.__version__)')"
  echo "transformers=$(python -c 'import transformers; print(transformers.__version__)')"
  echo "torch=$(python -c 'import torch; print(torch.__version__)')"
  echo "decoder_base=${MODEL_NAME}"
  echo "encoder=${TEXT_ENCODER_NAME}"
  echo "max_source_len=${MAX_SOURCE_LEN} max_vision_len=${MAX_VISION_LEN}"
  echo "batch_size=8 seed=${SEED} max_samples=0"
} > "${OUT}/PROVENANCE"
cat "${OUT}/PROVENANCE"

# Именование клеток по ФАКТИЧЕСКОМУ источнику весов декодера, а не по жаргону.
cat > "${OUT}/LABELS" <<'EOF'
diag_pretrained_decoder_llmlingua_cr{CR}
    Text-input diagnostic. Decoder weights = HuggingFaceTB/SmolLM2-1.7B-Instruct
    exactly as published: eval_cvlm.py is invoked WITHOUT --checkpoint_path, so
    no CVLM checkpoint (whose decoder IS trained, 97.94% trainable params) is
    loaded. This cell is therefore NOT a "CVLM-trained decoder" cell; it is the
    published instruct decoder fed a compressed text document. Directly paired
    with runs/fresh_heldout/frozen_{first,oracle}_cr{CR}_dump.jsonl, which were
    produced with the identical invocation.

primary_sfttrunc_decoder_llmlingua_cr{CR}
    Truncated-SFT decoder, PRIMARY baseline. Weights = runs/sft_trunc{CR}_*,
    i.e. SmolLM2 fine-tuned on "document truncated to the cr budget + question".
    The decoder is used to the input format, so this cell isolates the quality
    of the compression, not format shock. Directly paired with
    runs/fresh_heldout/sfttrunc_cr{CR}_dump.jsonl (first-N truncation).
EOF
cat "${OUT}/LABELS"

# Общие флаги датасета: ОДИН И ТОТ ЖЕ набор для селекции, аудита и всех четырёх
# оценок — иначе idx в JSONL перестанет соответствовать строкам датасета.
# Массив, а не строка: "test[2100:5100]" — валидный glob, при word splitting
# его могло бы съесть pathname expansion.
COMMON=(
  --dataset_name sggetao/PwC --dataset_split "${SPLIT}"
  --model_name_or_path "${MODEL_NAME}"
  --text_encoder_name "${TEXT_ENCODER_NAME}"
  --vision_encoder_name "${VISION_ENCODER_NAME}"
  --num_pool_latents 4 --max_prompt_len 512 --max_answer_len 1024
  --max_vision_len "${MAX_VISION_LEN}" --max_source_len "${MAX_SOURCE_LEN}"
  --max_samples 0 --batch_size 8 --seed "${SEED}"
)

# --- фаза 1: селекция LLMLingua-2 при бюджете cr=8 и cr=16 --------------------
for CR in 8 16; do
  SEL="${OUT}/llmlingua_cr${CR}.jsonl"
  if [ ! -s "${SEL}" ]; then
    echo; echo "=== [1] LLMLingua-2 selection, budget_cr=${CR}"
    python tools/llmlingua_select.py \
      --dataset_name sggetao/PwC --dataset_split "${SPLIT}" \
      --decoder_tokenizer_name "${MODEL_NAME}" \
      --encoder_tokenizer_name "${TEXT_ENCODER_NAME}" \
      --max_prompt_len 512 --max_answer_len 1024 --max_source_len "${MAX_SOURCE_LEN}" \
      --max_samples 0 \
      --budget_cr "${CR}" --max_vision_len "${MAX_VISION_LEN}" \
      --llmlingua_model "${LINGUA_MODEL}" --device cuda \
      --calibrate_rounds 2 --overshoot 1.10 --seed "${SEED}" \
      --output_jsonl "${SEL}" \
      --stats_json "${OUT}/llmlingua_cr${CR}_stats.json"
  fi
done

# --- фаза 2: аудит селекции (CPU) — до единой секунды GPU на оценках ----------
for CR in 8 16; do
  echo; echo "=== [2] audit selection cr=${CR}"
  python tools/audit_selection_jsonl.py \
    --jsonl "${OUT}/llmlingua_cr${CR}.jsonl" \
    --budget_cr "${CR}" --max_vision_len "${MAX_VISION_LEN}" \
    --dataset_name sggetao/PwC --dataset_split "${SPLIT}" \
    --decoder_tokenizer_name "${MODEL_NAME}" \
    --encoder_tokenizer_name "${TEXT_ENCODER_NAME}" \
    --max_prompt_len 512 --max_answer_len 1024 --max_source_len "${MAX_SOURCE_LEN}" \
    --max_samples 0 \
    --out_json "${OUT}/llmlingua_cr${CR}_audit.json"

  # Бюджет посчитан в токенах базового SmolLM2, а в режиме sft датасет
  # ре-токенизирует selected_text ТОКЕНИЗАТОРОМ SFT-директории. Разойдись они —
  # "равный бюджет" был бы равным только на бумаге.
  SFT_DIR="$(ls -d /home/aburlova/cvlm/runs/sft_trunc${CR}_*/ | sort | tail -1)"
  echo "=== truncated-SFT cr=${CR}: ${SFT_DIR}"
  SFT_DIR="${SFT_DIR}" SEL="${OUT}/llmlingua_cr${CR}.jsonl" BASE="${MODEL_NAME}" python - <<'PY'
import json, os
from transformers import AutoTokenizer
base = AutoTokenizer.from_pretrained(os.environ["BASE"], use_fast=False)
sft = AutoTokenizer.from_pretrained(os.environ["SFT_DIR"], use_fast=False)
bad = n = 0
with open(os.environ["SEL"]) as f:
    for line in f:
        rec = json.loads(line)
        t = rec["selected_text"]
        n += 1
        if base(t, add_special_tokens=False)["input_ids"] != sft(t, add_special_tokens=False)["input_ids"]:
            bad += 1
if bad:
    raise SystemExit(f"FATAL: SFT tokenizer disagrees with the base tokenizer on {bad}/{n} selections")
print(f"tokenizer check OK: base == SFT on all {n} selections")
PY
done

# --- фаза 3: четыре оценки = 2 cr x 2 декодера --------------------------------
for CR in 8 16; do
  SEL="${OUT}/llmlingua_cr${CR}.jsonl"

  # (а) предобученный SmolLM2-1.7B-Instruct, БЕЗ --checkpoint_path
  J="${OUT}/diag_pretrained_decoder_llmlingua_cr${CR}.json"
  if [ ! -s "${J}" ]; then
    echo; echo "=== [3a] pretrained SmolLM2 (text-input diagnostic) + LLMLingua-2, cr=${CR}"
    python src/eval_cvlm.py --mode baseline_llm_full "${COMMON[@]}" --compression_rate 1 \
      --doc_truncation file --doc_selection_file "${SEL}" \
      --output_json "${J}" \
      --dump_per_sample "${OUT}/diag_pretrained_decoder_llmlingua_cr${CR}_dump.jsonl"
  fi

  # (б) truncated-SFT декодер того же cr — ОСНОВНАЯ клетка
  SFT_DIR="$(ls -d /home/aburlova/cvlm/runs/sft_trunc${CR}_*/ | sort | tail -1)"
  J="${OUT}/primary_sfttrunc_decoder_llmlingua_cr${CR}.json"
  if [ ! -s "${J}" ]; then
    echo; echo "=== [3b] truncated-SFT(cr=${CR}) decoder (PRIMARY) + LLMLingua-2: ${SFT_DIR}"
    python src/eval_cvlm.py --mode sft --sft_model_path "${SFT_DIR}" "${COMMON[@]}" --compression_rate 1 \
      --doc_truncation file --doc_selection_file "${SEL}" \
      --output_json "${J}" \
      --dump_per_sample "${OUT}/primary_sfttrunc_decoder_llmlingua_cr${CR}_dump.jsonl"
  fi
done

# --- фаза 4: парный кластерный bootstrap по документам ------------------------
# Первично NLL; PPL — экспонента АГРЕГИРОВАННОГО NLL. Референсные клетки того
# же сплита берутся из ${REF}; отсутствующие просто не попадут в отчёт.
for CR in 8 16; do
  echo; echo "=== [4] paired document-clustered bootstrap, cr=${CR}"
  python tools/llmlingua_report.py \
    --clusters_jsonl "${OUT}/llmlingua_cr${CR}.jsonl" \
    --cell "llmlingua_sft=${OUT}/primary_sfttrunc_decoder_llmlingua_cr${CR}_dump.jsonl" \
    --cell "llmlingua_pre=${OUT}/diag_pretrained_decoder_llmlingua_cr${CR}_dump.jsonl" \
    --cell "first_sft=${REF}/sfttrunc_cr${CR}_dump.jsonl" \
    --cell "first_pre=${REF}/frozen_first_cr${CR}_dump.jsonl" \
    --cell "oracle_pre=${REF}/frozen_oracle_cr${CR}_dump.jsonl" \
    --cell "cvlm=${REF}/compressor_cr${CR}_dump.jsonl" \
    --contrast llmlingua_sft:first_sft \
    --contrast llmlingua_pre:first_pre \
    --contrast llmlingua_pre:oracle_pre \
    --contrast cvlm:llmlingua_sft \
    --contrast cvlm:llmlingua_pre \
    --n_boot 5000 --seed "${SEED}" \
    --out_json "${OUT}/bootstrap_cr${CR}.json"
done

sha256sum -c "${OUT}/CODE_SHA256"
echo "=== LLMLINGUA BASELINE DONE: ${OUT}"
