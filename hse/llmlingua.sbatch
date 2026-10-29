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
# РАВНЫЙ БЮДЖЕТ (и почему "<=", а не "=="). Бюджет —
#     V_i = max(min(ceil(min(L_ModernBERT_i, MAX_SOURCE_LEN)/CR), MAX_VISION_LEN), 1)
# токенов ТОКЕНИЗАТОРА ДЕКОДЕРА (SmolLM2); формула побитово та же, что в
# CvlmTrainDataset._truncated_doc_ids. Ищем ПАРАМЕТР, строку НЕ обрезаем:
# LLMLingua отдаёт отобранные слова в порядке документа, и отрезание хвоста
# превратило бы опубликованный метод во front-biased отбор — фору, придуманную
# нами в свою же пользу.
# Родной knob LLMLingua (rate) доходит до её же правила отбора только через
# q = int(100*(1-rate)+1) — ЦЕЛЫЙ перцентиль; на этих документах один шаг
# перцентиля двигает ~5 слов ≈ 7 токенов декодера, т.е. поиск по rate
# систематически недобирает 10-20% бюджета. Поэтому tools/llmlingua_select.py
# гоняет ЕЁ ЖЕ правило (per-chunk порог по вероятностям слов) с НЕПРЕРЫВНЫМ q,
# бисекцией до максимальной строки, влезающей в V_i. Совпадение с публичным
# API проверяется побайтово на 25 документах ДО начала отбора (--fidelity_*).
# Точное равенство V_i всё равно не гарантировано: единица отбора — целое слово
# ценой 1..k токенов. Фатально проверяется ТОЛЬКО <= V_i (бейзлайн не может
# получить слотов больше, чем CVLM); фактический недобор (доля и средняя
# величина) измеряется и идёт в отчёт как есть.
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
#            (runs/sft_trunc${CR}_*). Он адаптирован к КОРОТКОМУ текстовому
#            входу, но обучался на first-N, не на выходах LLMLingua. Поэтому
#            чистый контраст этой клетки — с first-N под ТЕМ ЖЕ декодером;
#            сравнение с CVLM является system-level, а не изоляцией одного
#            фактора. Selection-specific distribution shift не скрываем.
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
# Запуск (снапшот обязателен):
#   CVLM_SNAPSHOT_ROOT=/home/aburlova/cvlm_<immutable_id> sbatch hse/llmlingua.sbatch
# ============================================================================

set -euo pipefail
mkdir -p /home/aburlova/cvlm/runs/_logs

module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/cvlm

LLMLINGUA_PKG_VERSION="$(python -c 'from importlib.metadata import version; print(version("llmlingua"))')"
test "${LLMLINGUA_PKG_VERSION}" = "0.2.2" || {
  echo "ERROR: selector mirrors llmlingua==0.2.2, found ${LLMLINGUA_PKG_VERSION}"; exit 1; }

# No stale fallback: this baseline depends on the audited selector below and
# on llmlingua==0.2.2 private details. Running an older snapshot silently is
# worse than failing before allocation.
SNAP="${CVLM_SNAPSHOT_ROOT:?submit with CVLM_SNAPSHOT_ROOT=/path/to/immutable/snapshot}"
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
# Пути truncated-SFT декодеров ЗАФИКСИРОВАНЫ явно. Никаких "ls | tail -1":
# так молча подхватился бы более поздний упавший/частичный прогон, и клетка
# основного бейзлайна поехала бы без единого сообщения об ошибке.
SFT8_DIR="${SFT8_DIR:-/home/aburlova/cvlm/runs/sft_trunc8_20260730_072938}"
SFT16_DIR="${SFT16_DIR:-/home/aburlova/cvlm/runs/sft_trunc16_20260730_072938}"
check_sft () {  # $1 = каталог, $2 = ожидаемый cr (только для сообщения)
  local d="$1" cr="$2" f
  test -d "${d}" || { echo "ERROR: truncated-SFT dir for cr=${cr} not found: ${d}"; exit 1; }
  for f in config.json model.safetensors tokenizer_config.json vocab.json merges.txt; do
    test -s "${d}/${f}" || { echo "ERROR: ${d} is not a complete HF model dir (missing/empty ${f})"; exit 1; }
  done
  grep -q "\[train_sft\] Done\." "${d}/pipeline.log" || {
    echo "ERROR: ${d}/pipeline.log has no '[train_sft] Done.' — training did not finish"; exit 1; }
  echo "truncated-SFT cr=${cr}: ${d} — complete"
}
check_sft "${SFT8_DIR}" 8
check_sft "${SFT16_DIR}" 16
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
  echo "llmlingua_pkg_version=${LLMLINGUA_PKG_VERSION}"
  echo "numpy=$(python -c 'import numpy; print(numpy.__version__)')"
  echo "transformers=$(python -c 'import transformers; print(transformers.__version__)')"
  echo "torch=$(python -c 'import torch; print(torch.__version__)')"
  echo "decoder_base=${MODEL_NAME}"
  echo "encoder=${TEXT_ENCODER_NAME}"
  echo "max_source_len=${MAX_SOURCE_LEN} max_vision_len=${MAX_VISION_LEN}"
  echo "batch_size=8 seed=${SEED} max_samples=0"
  echo "sft_trunc8_dir=${SFT8_DIR}"
  echo "sft_trunc16_dir=${SFT16_DIR}"
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
    Truncation-adapted decoder, primary task-adapted control. Weights =
    runs/sft_trunc{CR}_*: SmolLM2 fine-tuned on "first-N document truncated to
    the cr budget + question", NOT on LLMLingua outputs. This controls generic
    short-text/task adaptation but not LLMLingua-specific selection shift.
    Its clean selector contrast is with runs/fresh_heldout/sfttrunc_cr{CR}_dump.jsonl
    (first-N under the identical decoder); CVLM comparisons are system-level.
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
  STATS="${OUT}/llmlingua_cr${CR}_stats.json"
  # Both files are atomically published by llmlingua_select.py. Requiring the
  # pair prevents a preemption from turning a partial artefact into a silently
  # reused baseline on resubmission.
  if [ ! -s "${SEL}" ] || [ ! -s "${STATS}" ]; then
    echo; echo "=== [1] LLMLingua-2 selection, budget_cr=${CR}"
    python tools/llmlingua_select.py \
      --dataset_name sggetao/PwC --dataset_split "${SPLIT}" \
      --decoder_tokenizer_name "${MODEL_NAME}" \
      --encoder_tokenizer_name "${TEXT_ENCODER_NAME}" \
      --max_prompt_len 512 --max_answer_len 1024 --max_source_len "${MAX_SOURCE_LEN}" \
      --max_samples 0 \
      --budget_cr "${CR}" --max_vision_len "${MAX_VISION_LEN}" \
      --llmlingua_model "${LINGUA_MODEL}" --device cuda \
      --search_mode threshold --search_iters 32 --fidelity_check_docs 25 \
      --seed "${SEED}" \
      --output_jsonl "${SEL}" \
      --stats_json "${STATS}"
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
    --max_samples 0 --expect_rows 3000 --expect_docs 205 \
    --out_json "${OUT}/llmlingua_cr${CR}_audit.json"

  # Бюджет посчитан в токенах базового SmolLM2, а в режиме sft датасет
  # ре-токенизирует selected_text ТОКЕНИЗАТОРОМ SFT-директории. Разойдись они —
  # "равный бюджет" был бы равным только на бумаге.
  case "${CR}" in 8) SFT_DIR="${SFT8_DIR}" ;; 16) SFT_DIR="${SFT16_DIR}" ;; *) echo "no SFT dir for cr=${CR}"; exit 1 ;; esac
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
    python src/eval_cvlm.py --mode baseline_llm_full "${COMMON[@]}" --compression_rate "${CR}" \
      --doc_truncation file --doc_selection_file "${SEL}" \
      --output_json "${J}" \
      --dump_per_sample "${OUT}/diag_pretrained_decoder_llmlingua_cr${CR}_dump.jsonl"
  fi

  # (б) truncated-SFT декодер того же cr — ОСНОВНАЯ клетка
  case "${CR}" in 8) SFT_DIR="${SFT8_DIR}" ;; 16) SFT_DIR="${SFT16_DIR}" ;; *) echo "no SFT dir for cr=${CR}"; exit 1 ;; esac
  J="${OUT}/primary_sfttrunc_decoder_llmlingua_cr${CR}.json"
  if [ ! -s "${J}" ]; then
    echo; echo "=== [3b] truncated-SFT(cr=${CR}) decoder (PRIMARY) + LLMLingua-2: ${SFT_DIR}"
    python src/eval_cvlm.py --mode sft --sft_model_path "${SFT_DIR}" "${COMMON[@]}" --compression_rate "${CR}" \
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
    --expect_rows 3000 --expect_docs 205 \
    --cell "llmlingua_sft=${OUT}/primary_sfttrunc_decoder_llmlingua_cr${CR}_dump.jsonl" \
    --cell "llmlingua_pre=${OUT}/diag_pretrained_decoder_llmlingua_cr${CR}_dump.jsonl" \
    --ref_cell "first_sft=${REF}/sfttrunc_cr${CR}_dump.jsonl" \
    --ref_cell "first_pre=${REF}/frozen_first_cr${CR}_dump.jsonl" \
    --ref_cell "oracle_pre=${REF}/frozen_oracle_cr${CR}_dump.jsonl" \
    --ref_cell "cvlm=${REF}/compressor_cr${CR}_dump.jsonl" \
    --contrast llmlingua_sft:first_sft \
    --contrast llmlingua_pre:first_pre \
    --contrast llmlingua_pre:oracle_pre \
    --contrast cvlm:llmlingua_sft \
    --contrast cvlm:llmlingua_pre \
    --n_boot 5000 --seed "${SEED}" \
    --out_json "${OUT}/bootstrap_cr${CR}.json"
done

sha256sum -c "${OUT}/CODE_SHA256"
touch "${OUT}/DONE"
echo "=== LLMLINGUA BASELINE DONE: ${OUT}"
