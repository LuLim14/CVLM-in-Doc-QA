#!/usr/bin/env python3
"""Собрать eval-JSONы веток абляции в главную таблицу статьи.

Вход: директории ранов (abl_vit_*, abl_randomvit_*, ...) с файлами
eval_cvlm_native_step<S>.json (native-cr евалы всех чекпоинтов) и
eval_baseline_llm*.json. Выход: tidy CSV + два LaTeX-вида:
  equal-compute        — метрики на чекпоинте ГРАНИЦЫ каждой стадии (одинаковый
                         step у всех веток при общем CR_SCHEDULE);
  convergence-adjusted — лучший по PPL чекпоинт ВНУТРИ стадии у каждой ветки.

Использование:
  python tools/aggregate_results.py --runs_root /home/aburlova/cvlm/runs \
      --schedule "1:1500,2:3000,4:4500,8:6000,16:0" --out_prefix paper_table
"""

import argparse
import glob
import json
import os
import re

METRIC_KEYS = {  # fuzzy-мэппинг: в JSON встречаются разные имена
    "ppl": ("perplexity", "ppl", "eval_perplexity"),
    "ta": ("token_accuracy", "token_acc", "tok_acc"),
    "rougeL": ("rougeL", "rouge_l", "rougeL_f", "rougeLsum"),
    "bleu4": ("bleu4", "bleu", "sacrebleu"),
}
BRANCH_ORDER = ["vit", "frozenvit", "randomvit", "novit"]
BRANCH_LABEL = {
    "vit": "pretrained ViT",
    "frozenvit": "frozen ViT",
    "randomvit": "random-init ViT",
    "novit": "no ViT",
}


def get_metric(d, name):
    for k in METRIC_KEYS[name]:
        if k in d and d[k] is not None:
            return float(d[k])
        for kk in d:  # на уровень глубже (например d["metrics"])
            if isinstance(d[kk], dict) and k in d[kk] and d[kk][k] is not None:
                return float(d[kk][k])
    return None


def parse_schedule(s):
    stages = []  # [(cr, end_step)]
    for part in s.split(","):
        cr, end = part.split(":")
        stages.append((int(cr), int(end)))
    return stages


def native_cr_for_step(stages, step, total_steps):
    for cr, end in stages:
        if end == 0 or step <= end:
            return cr
    return stages[-1][0]


def stage_bounds(stages, total_steps):
    out, prev = {}, 0
    for cr, end in stages:
        hi = end if end > 0 else total_steps
        out[cr] = (prev + 1, hi)
        prev = hi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--schedule", default="1:1500,2:3000,4:4500,8:6000,16:0")
    ap.add_argument("--stamp", default="", help="фильтр по штампу запуска (например 20260729_171933)")
    ap.add_argument("--total_steps", type=int, default=10**9)
    ap.add_argument("--out_prefix", default="paper_table")
    args = ap.parse_args()

    stages = parse_schedule(args.schedule)
    rows = []  # branch, step, cr, ppl, ta, rougeL, bleu4, path
    for br in BRANCH_ORDER:
        pat = os.path.join(args.runs_root, f"abl_{br}_*{args.stamp}*")
        run_dirs = sorted(glob.glob(pat))
        if not run_dirs:
            print(f"warn: нет ранов {pat}")
            continue
        rd = run_dirs[-1]
        for f in glob.glob(os.path.join(rd, "eval_cvlm*_step*.json")):
            m = re.search(r"_step(\d+)\.json$", f)
            if not m:
                continue
            step = int(m.group(1))
            with open(f) as fh:
                d = json.load(fh)
            cr = d.get("compression_rate") or native_cr_for_step(stages, step, args.total_steps)
            rows.append(dict(branch=br, step=step, cr=int(cr),
                             ppl=get_metric(d, "ppl"), ta=get_metric(d, "ta"),
                             rougeL=get_metric(d, "rougeL"), bleu4=get_metric(d, "bleu4"),
                             path=f))
    if not rows:
        raise SystemExit("Ни одного eval-JSON не найдено — рано или пути не те.")

    with open(args.out_prefix + ".csv", "w") as f:
        cols = ["branch", "step", "cr", "ppl", "ta", "rougeL", "bleu4", "path"]
        f.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda r: (r["branch"], r["step"])):
            f.write(",".join(str(r[c]) if r[c] is not None else "" for c in cols) + "\n")
    print(f"tidy: {args.out_prefix}.csv  ({len(rows)} evals)")

    max_step = max(r["step"] for r in rows)
    bounds = stage_bounds(stages, max_step)

    def cell(r):
        if r is None:
            return "--"
        parts = [f"{r['ppl']:.2f}" if r["ppl"] else "--",
                 f"{r['ta']:.3f}" if r["ta"] else "--",
                 f"{r['rougeL']:.3f}" if r["rougeL"] else "--"]
        return " / ".join(parts)

    crs = [cr for cr, _ in stages]
    for view in ("equal_compute", "convergence_adjusted"):
        lines = [
            "% " + view + ": PPL / TokAcc / ROUGE-L",
            r"\begin{tabular}{l" + "c" * len(crs) + "}",
            r"\toprule",
            "compressor & " + " & ".join(f"cr={c}" for c in crs) + r" \\",
            r"\midrule",
        ]
        for br in BRANCH_ORDER:
            cells = []
            for cr in crs:
                lo, hi = bounds[cr]
                cand = [r for r in rows if r["branch"] == br and r["cr"] == cr and lo <= r["step"] <= hi]
                if not cand:
                    cells.append("--")
                    continue
                if view == "equal_compute":
                    pick = max(cand, key=lambda r: r["step"])  # конец стадии
                else:
                    with_ppl = [r for r in cand if r["ppl"] is not None]
                    pick = min(with_ppl, key=lambda r: r["ppl"]) if with_ppl else max(cand, key=lambda r: r["step"])
                cells.append(cell(pick))
            lines.append(BRANCH_LABEL[br] + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        out = f"{args.out_prefix}_{view}.tex"
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"LaTeX: {out}")


if __name__ == "__main__":
    main()
