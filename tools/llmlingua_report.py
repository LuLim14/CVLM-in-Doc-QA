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
filtered-dataset idx space.

Nothing is silently dropped. A quiet ``set.intersection`` over the available
idx would let a half-written dump shrink the analysis set and still print a
confident number, so instead:

  * every ``--cell`` is REQUIRED: missing file, duplicate idx, or an idx set
    that differs from the selection's is fatal;
  * ``--ref_cell`` is an optional comparison arm — absent is fine, but present
    means it must pass the identical checks;
  * ``answer_tokens`` must agree row-by-row across every loaded arm (the
    answer side of the batch is identical by construction; a disagreement
    means the arms were not evaluated on the same data);
  * ``--expect_rows`` / ``--expect_docs`` pin the analysis set size.

Example::

    python tools/llmlingua_report.py \
        --clusters_jsonl runs/llmlingua_baseline/llmlingua_cr8.jsonl \
        --expect_rows 3000 --expect_docs 205 \
        --cell llmlingua_sft=runs/llmlingua_baseline/primary_..._cr8_dump.jsonl \
        --ref_cell first_sft=runs/fresh_heldout/sfttrunc_cr8_dump.jsonl \
        --contrast llmlingua_sft:first_sft \
        --out_json runs/llmlingua_baseline/bootstrap_cr8.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np


def load_dump(path: str) -> Dict[int, dict]:
    """Load a --dump_per_sample JSONL, failing loudly on duplicate idx."""
    rows: Dict[int, dict] = {}
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            i = int(r["idx"])
            if i in rows:
                raise SystemExit(f"FATAL {path}:{line_no}: duplicate idx {i}")
            rows[i] = r
    return rows


def per_doc_sums(
    rows: Dict[int, dict], docs: List[str], by_doc: Dict[str, List[int]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-document (nll_nats, answer_tokens, correct_tokens) sums.

    The bootstrap resamples documents, and every statistic here is a ratio of
    sums, so collapsing each document to three numbers once makes a resample a
    205-element gather instead of a 3000-element one — same estimator, ~15x
    less work, and it keeps 5000 resamples cheap enough to run inline.
    """
    nll = np.zeros(len(docs), dtype=np.float64)
    tok = np.zeros(len(docs), dtype=np.float64)
    cor = np.zeros(len(docs), dtype=np.float64)
    for d, doc in enumerate(docs):
        for i in by_doc[doc]:
            r = rows[i]
            nll[d] += float(r["answer_nll_nats"])
            tok[d] += float(r["answer_tokens"])
            cor[d] += float(r.get("correct_tokens", 0))
    return nll, tok, cor


def ratio(num: np.ndarray, den: np.ndarray) -> float:
    return float(num.sum() / max(den.sum(), 1.0))


def boot_ratio(num: np.ndarray, den: np.ndarray, boots: np.ndarray) -> np.ndarray:
    return num[boots].sum(axis=1) / np.maximum(den[boots].sum(axis=1), 1.0)


def ci(vals: np.ndarray) -> Tuple[float, float]:
    v = np.sort(vals)
    n = v.size
    return float(v[int(0.025 * n)]), float(v[int(0.975 * n)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired cluster bootstrap over documents")
    ap.add_argument("--clusters_jsonl", required=True, help="Selection JSONL carrying idx -> source_sha256.")
    ap.add_argument("--cell", action="append", default=[], metavar="NAME=PATH", help="Required arm.")
    ap.add_argument("--ref_cell", action="append", default=[], metavar="NAME=PATH", help="Optional comparison arm.")
    ap.add_argument("--contrast", action="append", default=[], metavar="A:B", help="Reports NLL(A)-NLL(B).")
    ap.add_argument("--expect_rows", type=int, default=0)
    ap.add_argument("--expect_docs", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    # --- analysis set: documents and rows come from the selection file --------
    by_doc: Dict[str, List[int]] = {}
    sel_set: set = set()
    with open(args.clusters_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = str(r.get("source_sha256", ""))
            if not key:
                raise SystemExit(f"FATAL {args.clusters_jsonl}: no source_sha256 field")
            i = int(r["idx"])
            if i in sel_set:
                raise SystemExit(f"FATAL {args.clusters_jsonl}: duplicate idx {i}")
            sel_set.add(i)
            by_doc.setdefault(key, []).append(i)
    docs = sorted(by_doc)
    idxs_all = [i for d in docs for i in by_doc[d]]
    print(f"analysis set: {len(idxs_all)} rows in {len(docs)} documents (from {args.clusters_jsonl})")
    if args.expect_rows and len(idxs_all) != args.expect_rows:
        raise SystemExit(f"FATAL: {len(idxs_all)} rows != expected {args.expect_rows}")
    if args.expect_docs and len(docs) != args.expect_docs:
        raise SystemExit(f"FATAL: {len(docs)} documents != expected {args.expect_docs}")

    # --- arms ----------------------------------------------------------------
    cells: Dict[str, Dict[int, dict]] = {}

    def add(spec: str, required: bool) -> None:
        name, _, path = spec.partition("=")
        if not os.path.exists(path):
            if required:
                raise SystemExit(f"FATAL: required cell {name} missing: {path}")
            print(f"[absent]  {name}: {path}")
            return
        rows = load_dump(path)
        if set(rows) != sel_set:
            missing = len(sel_set - set(rows))
            extra = len(set(rows) - sel_set)
            raise SystemExit(
                f"FATAL: cell {name} idx set differs from the selection "
                f"({missing} missing, {extra} extra): {path}"
            )
        cells[name] = rows
        print(f"[loaded]  {name}: {len(rows)} rows  ({path})")

    for spec in args.cell:
        add(spec, required=True)
    for spec in args.ref_cell:
        add(spec, required=False)
    if not cells:
        raise SystemExit("FATAL: no cells to report")

    # answer_tokens must agree row-by-row: the answer side is identical across
    # arms by construction, so a mismatch means different data, not a result.
    names = sorted(cells)
    base = names[0]
    for name in names[1:]:
        bad: List[Tuple[int, int, int]] = [
            (i, int(cells[base][i]["answer_tokens"]), int(cells[name][i]["answer_tokens"]))
            for i in idxs_all
            if int(cells[base][i]["answer_tokens"]) != int(cells[name][i]["answer_tokens"])
        ]
        if bad:
            head = ", ".join(f"idx={i}: {x} vs {y}" for i, x, y in bad[:5])
            raise SystemExit(
                f"FATAL: answer_tokens disagree between {base} and {name} on {len(bad)} rows ({head})"
            )
    print(f"answer_tokens identical across all {len(names)} arms")

    rng = np.random.default_rng(args.seed)
    boots = rng.integers(0, len(docs), size=(args.n_boot, len(docs)))
    sums = {name: per_doc_sums(cells[name], docs, by_doc) for name in names}

    out: Dict[str, object] = {
        "clusters_jsonl": args.clusters_jsonl,
        "n_docs": len(docs),
        "n_rows": len(idxs_all),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "cells": {},
        "contrasts": {},
    }
    print("\n=== CELLS (NLL primary; PPL = exp of the AGGREGATED NLL) ===")
    for name in names:
        rows = cells[name]
        s_nll, s_tok, s_cor = sums[name]
        nll = ratio(s_nll, s_tok)
        lo, hi = ci(boot_ratio(s_nll, s_tok, boots))
        ta = ratio(s_cor, s_tok)
        ta_lo, ta_hi = ci(boot_ratio(s_cor, s_tok, boots))
        print(
            f"  {name:22s} NLL={nll:.4f} [{lo:.4f},{hi:.4f}]  PPL={math.exp(nll):8.4f} "
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
        print("\n=== PAIRED CONTRASTS (same resampled documents in both arms) ===")
    for spec in args.contrast:
        a, _, b = spec.partition(":")
        if a not in cells or b not in cells:
            print(f"  {spec:34s} — skipped (optional arm {a if a not in cells else b} absent)")
            continue
        an, at, ac = sums[a]
        bn, bt, bc = sums[b]
        d_nll = ratio(an, at) - ratio(bn, bt)
        d_ta = ratio(ac, at) - ratio(bc, bt)
        lo, hi = ci(boot_ratio(an, at, boots) - boot_ratio(bn, bt, boots))
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
