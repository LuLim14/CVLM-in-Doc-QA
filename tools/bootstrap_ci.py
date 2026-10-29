#!/usr/bin/env python3
"""Percentile-bootstrap CIs over per-sample eval dumps (eval_cvlm.py --dump_per_sample).

Single system:
    python tools/bootstrap_ci.py runA.jsonl
Paired comparison (aligned by "idx"; reports deltas B-A + p-values):
    python tools/bootstrap_ci.py runA.jsonl runB.jsonl

Metrics: PPL = exp(sum nll / sum tokens), token_accuracy = sum correct / sum tokens
(both pooled, recomputed per resample — matches eval_cvlm.py aggregation),
mean per-sample ROUGE-L (if present), corpus BLEU-4 via sacrebleu (if predictions
present and sacrebleu is importable; BLEU is resampled at corpus level).
"""

import argparse
import json
import math
import sys

import numpy as np


def load_dump(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        sys.exit(f"empty dump: {path}")
    return {r["idx"]: r for r in rows}


def pooled_metrics(rows, idxs):
    nll = sum(rows[i]["answer_nll_nats"] for i in idxs)
    tok = sum(rows[i]["answer_tokens"] for i in idxs)
    cor = sum(rows[i]["correct_tokens"] for i in idxs)
    out = {
        "ppl": math.exp(min(nll / max(tok, 1), 30.0)),
        "token_accuracy": cor / max(tok, 1),
    }
    if all("rougeL" in rows[i] and rows[i]["rougeL"] is not None for i in idxs[:5]):
        try:
            out["rougeL"] = float(np.mean([rows[i]["rougeL"] for i in idxs]))
        except (KeyError, TypeError):
            pass
    return out


def corpus_bleu(rows, idxs):
    try:
        import sacrebleu
    except ImportError:
        return None
    try:
        preds = [rows[i]["prediction"] for i in idxs]
        refs = [[rows[i]["reference"] for i in idxs]]
    except KeyError:
        return None
    return sacrebleu.corpus_bleu(preds, refs).score


def ci(vals, alpha=0.05):
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_a")
    ap.add_argument("dump_b", nargs="?", default=None)
    ap.add_argument("--n_resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with_bleu", action="store_true", help="also bootstrap corpus BLEU (slow)")
    args = ap.parse_args()

    A = load_dump(args.dump_a)
    B = load_dump(args.dump_b) if args.dump_b else None
    if B is not None:
        common = sorted(set(A) & set(B))
        dropped = (len(A) - len(common), len(B) - len(common))
        if any(dropped):
            print(f"warn: unaligned idx dropped: A-{dropped[0]}, B-{dropped[1]}")
    else:
        common = sorted(A)
    n = len(common)
    print(f"samples: {n}, resamples: {args.n_resamples}")

    rng = np.random.default_rng(args.seed)
    idx_arr = np.asarray(common)

    point_a = pooled_metrics(A, common)
    if args.with_bleu:
        b = corpus_bleu(A, common)
        if b is not None:
            point_a["bleu4"] = b
    boots_a = {k: [] for k in point_a}
    boots_d = None
    point_b = point_d = None
    if B is not None:
        point_b = pooled_metrics(B, common)
        if args.with_bleu and "bleu4" in point_a:
            bb = corpus_bleu(B, common)
            if bb is not None:
                point_b["bleu4"] = bb
        shared = set(point_a) & set(point_b)
        point_d = {k: point_b[k] - point_a[k] for k in shared}
        boots_d = {k: [] for k in shared}

    for _ in range(args.n_resamples):
        sample = list(idx_arr[rng.integers(0, n, size=n)])
        ma = pooled_metrics(A, sample)
        if args.with_bleu and "bleu4" in point_a:
            v = corpus_bleu(A, sample)
            if v is not None:
                ma["bleu4"] = v
        for k, v in ma.items():
            if k in boots_a:
                boots_a[k].append(v)
        if B is not None:
            mb = pooled_metrics(B, sample)
            if args.with_bleu and "bleu4" in (point_b or {}):
                v = corpus_bleu(B, sample)
                if v is not None:
                    mb["bleu4"] = v
            for k in boots_d:
                if k in ma and k in mb:
                    boots_d[k].append(mb[k] - ma[k])

    print(f"\n== {args.dump_a}")
    for k, v in point_a.items():
        lo, hi = ci(boots_a[k])
        print(f"  {k:16s} {v:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
    if B is not None:
        print(f"\n== {args.dump_b}")
        for k, v in point_b.items():
            print(f"  {k:16s} {v:.4f}")
        print("\n== paired delta (B - A)")
        for k, d in point_d.items():
            arr = np.asarray(boots_d[k])
            lo, hi = ci(arr)
            p = 2 * min((arr <= 0).mean(), (arr >= 0).mean())
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {k:16s} {d:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]  p~{p:.4f} {sig}")


if __name__ == "__main__":
    main()
