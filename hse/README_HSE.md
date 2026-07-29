# Запуск абляций CVLM на cHARISMa — runbook

Цель: собрать к 01–02.08 таблицы prior-абляции (pretrained/random/frozen/no-ViT)
и truncated-SFT бейзлайны для DocInsights 2026. Протокол всех веток един и
задан ТОЛЬКО в `protocol.env`.

## Порядок действий

```bash
# 0. Залить код на кластер (с Мака, из корня репозитория):
rsync -avz --exclude .git --exclude __pycache__ --exclude runs \
    ~/Desktop/CVLM/code/ hse-trail:/home/aburlova/cvlm/

# 1. ЛОГИН-нода: окружение (~15 мин, один раз)
ssh hse-trail "bash /home/aburlova/cvlm/hse/setup_env.sh"

# 2. ЛОГИН-нода: предзагрузка HF (модели+датасет, ~10 мин; упадёт, если у PwC нет split test!)
ssh hse-trail "bash /home/aburlova/cvlm/hse/prefetch.sh"

# 3. Смоук (gpu-ef-quick, ~1-2 ч): 4 ветки по 30 шагов + микро-eval + замер сек/шаг
ssh hse-trail "cd /home/aburlova/cvlm/hse && sbatch smoke.sbatch"
#    После завершения посмотреть batch_time в логе; если сек/шаг сильно > 12с —
#    ужать CR_SCHEDULE в protocol.env (одинаково для всех веток!) и только потом п.4.

# 4. Боевой запуск: 4 абляции + 2 truncated-SFT параллельно
ssh hse-trail "bash /home/aburlova/cvlm/hse/launch_all.sh"

# 5. Мониторинг (не чаще раза в ~минуту, батчить):
ssh hse-trail "squeue -u aburlova -o '%i %j %T %M %R'; tail -3 /home/aburlova/cvlm/runs/_logs/abl_vit_*.out"

# 6. Забрать результаты на Мак:
rsync -avz --include '*/' --include 'eval_*.json' --include 'metrics.csv' \
    --include 'pipeline.log' --include '*.png' --exclude '*' \
    hse-trail:/home/aburlova/cvlm/runs/ ~/Desktop/CVLM/runs_hse/
```

## Ветки

| job | флаги | что проверяет |
|---|---|---|
| abl_vit | (дефолт) | pretrained trainable ViT — референс |
| abl_randomvit | VIT_INIT=random | вклад ИМЕННО визуальных весов (RQ1) |
| abl_frozenvit | FREEZE_VIT=1 | пригоден ли приор без переписывания (RQ1b) |
| abl_novit | NO_VIT=1 | нужна ли capacity ViT вообще (RQ2) |
| sft_trunc8/16 | TRUNC_CR=8/16 | trained-truncation бейзлайн (RQ3) |
| sft_full (опц.) | TRUNC_CR=0 | ceiling |

## Тайминг и риски

- Укороченный протокол ~7.5k шагов ≈ 15–27 ч/ветку на A100 (оценка; смоук уточнит).
  Форс-чекпоинт на каждой границе стадии ⇒ даже упавшая на cr=16 ветка отдаёт cr≤8.
- Walltime 2 суток; если не влезает — resume:
  `sbatch --job-name=abl_vit --export=ALL,BRANCH=vit,RESUME=/home/aburlova/cvlm/runs/abl_vit_<stamp> train_branch.sbatch`
  (нюанс: mid-epoch resume перезапускает эпоху при сохранённом global_step — позиция куррикулума не теряется).
- Диск: ~11–12 GB на пару чекпоинтов, до ~100 GB на ветку. sbatch в конце сам
  чистит старые trainer_step_*.pt. Проверять: `du -sh /home/aburlova/cvlm/runs`.
- Если liger/triton не заведётся на compute-ноде: аварийно `DISABLE_LIGER=1` +
  BATCH_SIZE=16, GRAD_ACCUM=4 в protocol.env (для ВСЕХ веток одинаково).
- ВАЖНО для eval no-ViT ветки: флаг `--no_vit` обязан передаваться и в eval
  (run_full_pipeline.sh это делает автоматически через NO_VIT).

## Что дальше (после схода прогонов — отдельные скрипты, wave 2)

- truncation-сетка inference-time (first/last/random/oracle-window),
- cross-CR матрица (каждый стадийный чекпоинт × все cr, включая 32),
- bootstrap-CI + пер-сэмпловый дамп,
- eval truncated-SFT с усечённым документом (симметрия train/eval).
