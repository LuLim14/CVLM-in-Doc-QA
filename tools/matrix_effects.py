#!/usr/bin/env python3
"""Четыре эффекта матрицы precision×prior в ПАРНОМ NLL с кластерным bootstrap.

Первичная величина — разность NLL (нат/answer-токен): она аддитивна, поэтому
interaction определён корректно. PPL и отношение PPL приводятся как вторичные,
интерпретируемые числа (exp от NLL-разности).

Эффекты:
  depth      = NLL(Pool+12L Mixer)      - NLL(Pool+Project)        [в каждом режиме точности]
  prior      = NLL(Pretrained-ViT)      - NLL(random-init Mixer)   [в каждом режиме]
  precision  = NLL(fp32-мастера)        - NLL(legacy bf16)         [для random и для pretrained]
  interaction= prior_fp32 - prior_bf16   (помогает ли fp32 приору сильнее, чем random)

Bootstrap: ресэмплятся ДОКУМЕНТЫ (в PwC ~15 вопросов на документ, строки не независимы).

Запуск на кластере:
  python tools/matrix_effects.py --split "test[2100:5100]" \
      --dump legacy_random=/path/seed43_randomvit_cr16_dump.jsonl \
      --dump legacy_pre=/path/seed43_vit_cr16_dump.jsonl \
      --dump fp32_random=/path/fp32_randomvit/eval_cvlm_dump.jsonl \
      --dump fp32_pre=/path/fp32_vit/eval_cvlm_dump.jsonl \
      --dump pool=/path/seed43_novit_cr16_dump.jsonl
Отсутствующие клетки просто пропускаются с явным сообщением.
"""

import argparse
import hashlib
import json
import math
import os
import random


def load(path):
    rows = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["idx"]] = r
    return rows


def nll(rows, idxs):
    s = sum(rows[i]["answer_nll_nats"] for i in idxs)
    t = sum(rows[i]["answer_tokens"] for i in idxs)
    return s / max(t, 1)


def ta(rows, idxs):
    c = sum(rows[i].get("correct_tokens", 0) for i in idxs)
    t = sum(rows[i]["answer_tokens"] for i in idxs)
    return c / max(t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test[2100:5100]")
    ap.add_argument("--dump", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    dumps = {}
    for spec in args.dump:
        name, _, path = spec.partition("=")
        if not os.path.exists(path):
            print(f"[нет] {name}: {path}")
            continue
        dumps[name] = load(path)
        print(f"[есть] {name}: {len(dumps[name])} строк")
    if len(dumps) < 2:
        raise SystemExit("нужно минимум два дампа")

    from datasets import load_dataset

    ds = load_dataset("sggetao/PwC", split=args.split)
    by_doc = {}
    for i, text in enumerate(ds["input"]):
        by_doc.setdefault(hashlib.md5(text.encode()).hexdigest(), []).append(i)
    common = set.intersection(*[set(d) for d in dumps.values()])
    docs = [d for d, idxs in by_doc.items() if all(i in common for i in idxs)]
    idxs_all = [i for d in docs for i in by_doc[d]]
    print(f"\nобщих строк: {len(common)}, полных документов: {len(docs)}\n")

    rng = random.Random(args.seed)
    boots = [[i for _ in range(len(docs)) for i in by_doc[docs[rng.randrange(len(docs))]]]
             for _ in range(args.n_boot)]

    def contrast(a, b, label):
        if a not in dumps or b not in dumps:
            print(f"  {label:34s} — пропуск (нет {a if a not in dumps else b})")
            return None
        d_nll = nll(dumps[a], idxs_all) - nll(dumps[b], idxs_all)
        d_ta = ta(dumps[a], idxs_all) - ta(dumps[b], idxs_all)
        vals = sorted(nll(dumps[a], s) - nll(dumps[b], s) for s in boots)
        lo, hi = vals[int(0.025 * args.n_boot)], vals[int(0.975 * args.n_boot)]
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {label:34s} ΔNLL={d_nll:+.4f} [{lo:+.4f},{hi:+.4f}]{star} "
              f"(×PPL {math.exp(d_nll):.3f})  ΔTokAcc={d_ta:+.4f}")
        return dict(delta_nll=d_nll, lo=lo, hi=hi, ppl_ratio=math.exp(d_nll), delta_ta=d_ta)

    out = {"n_docs": len(docs), "n_rows": len(idxs_all), "split": args.split}
    print("=== DEPTH PENALTY (миксер минус Pool+Project) ===")
    out["depth_bf16"] = contrast("legacy_random", "pool", "legacy bf16")
    out["depth_fp32"] = contrast("fp32_random", "pool", "fp32 ViT-мастера")
    print("\n=== PRIOR EFFECT (pretrained минус random-init) ===")
    out["prior_bf16"] = contrast("legacy_pre", "legacy_random", "legacy bf16")
    out["prior_fp32"] = contrast("fp32_pre", "fp32_random", "fp32 ViT-мастера")
    print("\n=== PRECISION EFFECT (fp32 минус legacy) ===")
    out["prec_random"] = contrast("fp32_random", "legacy_random", "для random-init")
    out["prec_pre"] = contrast("fp32_pre", "legacy_pre", "для pretrained")
    print("\n=== INTERACTION (prior_fp32 − prior_bf16) ===")
    if out.get("prior_fp32") and out.get("prior_bf16"):
        pf, pb = dumps.get("fp32_pre"), dumps.get("legacy_pre")
        rf, rb = dumps.get("fp32_random"), dumps.get("legacy_random")
        point = ((nll(pf, idxs_all) - nll(rf, idxs_all))
                 - (nll(pb, idxs_all) - nll(rb, idxs_all)))
        vals = sorted(((nll(pf, s) - nll(rf, s)) - (nll(pb, s) - nll(rb, s))) for s in boots)
        lo, hi = vals[int(0.025 * args.n_boot)], vals[int(0.975 * args.n_boot)]
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {'interaction':34s} ΔΔNLL={point:+.4f} [{lo:+.4f},{hi:+.4f}]{star}")
        out["interaction"] = dict(delta=point, lo=lo, hi=hi)
    else:
        print("  interaction — пропуск: матрица неполная (нужны все четыре клетки)")
        out["interaction"] = None
    print("\n* = CI не накрывает ноль. Bootstrap кластерный по документам, "
          f"{args.n_boot} ресэмплов. NLL — первичная величина, PPL-отношение вторично.")
    if args.out_json:
        json.dump(out, open(args.out_json, "w"), indent=2)
        print(f"сохранено -> {args.out_json}")


if __name__ == "__main__":
    main()
