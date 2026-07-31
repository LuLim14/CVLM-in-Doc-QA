#!/usr/bin/env python3
"""Paired document-clustered bootstrap over eval per-sample dumps.

Primary quantity is NLL in nats per answer token, aggregated as
``sum(nll) / sum(tokens)`` over the resampled rows. PPL is reported as
``exp`` of that AGGREGATED NLL — never as a mean of per-sample perplexities,
which is a different (and wrong) estimator. Token accuracy is likewise
``sum(correct)/sum(tokens)``.

Rows are NOT independent: PwC asks ~15 questions about each document, so the
bootstrap resamples DOCUMENTS with replacement and takes all their rows
(cluster bootstrap), the same scheme as tools/matrix_effects.py. Contrasts are
paired: the same resampled row set is scored under both cells, so the CI is on
the difference, not on two independent means.

Document identity comes from the ``source_sha256`` field of a selection JSONL
(``tools/llmlingua_select.py``), whose idx space is by construction the eval's
filtered-dataset idx space. With ``--split`` instead, identity is recomputed
from the raw dataset — only valid when the length filter drops nothing.

Example::

    python tools/llmlingua_report.py \
        --clusters_jsonl runs/llmlingua_baseline/llmlingua_cr8.jsonl \
        --cell llmlingua_sft=runs/llmlingua_baseline/sfttrunc_llmlingua_cr8_dump.jsonl \
        --cell first_sft=runs/fresh_heldout/sfttrunc_cr8_dump.jsonl \
        --contrast llmlingua_sft:first_sft \
        --out_json runs/llmlingua_baseline/report_cr8.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from typing import Dict, List


def load_dump(path: str) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[int(r["idx"])] = r
    return rows


def agg_nll(rows: Dict[int, dict], idxs: List[int]) -> float:
    s = sum(rows[i]["answer_nll_nats"] for i in idxs)
    t = sum(rows[i]["answer_tokens"] for i in idxs)
    return s / max(t, 1)


def agg_ta(rows: Dict[int, dict], idxs: List[int]) -> float:
    c = sum(rows[i].get("correct_tokens", 0) for i in idxs)
    t = sum(rows[i]["answer_tokens"] for i in idxs)
    return c / max(t, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired cluster bootstrap over documents")
    ap.add_argument("--clusters_jsonl", default="", help="Selection JSONL carrying idx -> source_sha256.")
    ap.add_argument("--split", default="test[2100:5100]", help="Fallback cluster source (raw dataset).")
    ap.add_argument("--dataset_name", default="sggetao/PwC")
    ap.add_argument("--cell", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--contrast", action="append", default=[], metavar="A:B", help="Reports NLL(A)-NLL(B).")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    cells: Dict[str, Dict[int, dict]] = {}
    for spec in args.cell:
        name, _, path = spec.partition("=")
        if not os.path.exists(path):
            print(f"[missing] {name}: {path}")
            continue
        cells[name] = load_dump(path)
        print(f"[loaded]  {name}: {len(cells[name])} rows  ({path})")
    if not cells:
        raise SystemExit("no cells to report")

    by_doc: Dict[str, List[int]] = {}
    if args.clusters_jsonl:
        with open(args.clusters_jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = str(r.get("source_sha256", ""))
                if not key:
                    raise SystemExit(f"{args.clusters_jsonl}: no source_sha256; pass --split instead")
                by_doc.setdefault(key, []).append(int(r["idx"]))
    else:
        from datasets import load_dataset

        ds = load_dataset(args.dataset_name, split=args.split)
        for i, text in enumerate(ds["input"]):
            by_doc.setdefault(hashlib.sha256(str(text).encode()).hexdigest(), []).append(i)

    common = set.intersection(*[set(d) for d in cells.values()])
    docs = [d for d, idxs in by_doc.items() if all(i in common for i in idxs)]
    if not docs:
        raise SystemExit("no complete documents shared by all cells")
    idxs_all = [i for d in docs for i in by_doc[d]]
    print(f"\nrows shared by all cells: {len(common)}; complete documents: {len(docs)}; rows used: {len(idxs_all)}\n")

    rng = random.Random(args.seed)
    boots = [
        [i for _ in range(len(docs)) for i in by_doc[docs[rng.randrange(len(docs))]]]
        for _ in range(args.n_boot)
    ]

    out: Dict[str, object] = {
        "n_docs": len(docs),
        "n_rows": len(idxs_all),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "cells": {},
        "contrasts": {},
    }
    print("=== CELLS (NLL primary; PPL = exp of the AGGREGATED NLL) ===")
    for name, rows in cells.items():
        nll = agg_nll(rows, idxs_all)
        vals = sorted(agg_nll(rows, s) for s in boots)
        lo, hi = vals[int(0.025 * args.n_boot)], vals[int(0.975 * args.n_boot)]
        ta = agg_ta(rows, idxs_all)
        ta_vals = sorted(agg_ta(rows, s) for s in boots)
        ta_lo, ta_hi = ta_vals[int(0.025 * args.n_boot)], ta_vals[int(0.975 * args.n_boot)]
        print(
            f"  {name:34s} NLL={nll:.4f} [{lo:.4f},{hi:.4f}]  PPL={math.exp(nll):7.4f} "
            f"[{math.exp(lo):.4f},{math.exp(hi):.4f}]  TokAcc={ta:.4f} [{ta_lo:.4f},{ta_hi:.4f}]"
        )
        out["cells"][name] = {  # type: ignore[index]
            "nll_nats_per_token": nll,
            "nll_lo": lo,
            "nll_hi": hi,
            "ppl": math.exp(nll),
            "ppl_lo": math.exp(lo),
            "ppl_hi": math.exp(hi),
            "token_accuracy": ta,
            "token_accuracy_lo": ta_lo,
            "token_accuracy_hi": ta_hi,
            "n_rows": len(rows),
        }

    if args.contrast:
        print("\n=== PAIRED CONTRASTS (same resampled documents in both cells) ===")
    for spec in args.contrast:
        a, _, b = spec.partition(":")
        if a not in cells or b not in cells:
            print(f"  {spec:34s} — skipped (missing {a if a not in cells else b})")
            continue
        d_nll = agg_nll(cells[a], idxs_all) - agg_nll(cells[b], idxs_all)
        d_ta = agg_ta(cells[a], idxs_all) - agg_ta(cells[b], idxs_all)
        vals = sorted(agg_nll(cells[a], s) - agg_nll(cells[b], s) for s in boots)
        lo, hi = vals[int(0.025 * args.n_boot)], vals[int(0.975 * args.n_boot)]
        star = "*" if (lo > 0 or hi < 0) else " "
        print(
            f"  {a} - {b:24s} ΔNLL={d_nll:+.4f} [{lo:+.4f},{hi:+.4f}]{star} "
            f"(×PPL {math.exp(d_nll):.3f})  ΔTokAcc={d_ta:+.4f}"
        )
        out["contrasts"][f"{a}-{b}"] = {  # type: ignore[index]
            "delta_nll": d_nll,
            "lo": lo,
            "hi": hi,
            "ppl_ratio": math.exp(d_nll),
            "delta_token_accuracy": d_ta,
            "significant": bool(lo > 0 or hi < 0),
        }
    print("\n* = 95% cluster-bootstrap CI excludes zero. Clusters = documents.")
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"saved -> {args.out_json}")


if __name__ == "__main__":
    main()
