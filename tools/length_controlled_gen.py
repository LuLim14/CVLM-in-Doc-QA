#!/usr/bin/env python3
"""Length-controlled re-scoring of generative metrics from per-sample dumps.

WHY THIS EXISTS
---------------
`src/eval_cvlm.py --compute_generation_metrics` scores free-running generation
with `max_new_tokens=256` and no length normalisation.  The CVLM branch is
trained on targets tokenised with `add_special_tokens=False`
(`src/cvlm_dataset.py`), i.e. it never sees an EOS token and therefore never
learns to stop: it emits ~256 tokens for every question.  The truncated-SFT
baseline (`src/train_sft.py`) *does* append EOS and stops after ~1 sentence.
PwC references are short (~240 chars).  Raw ROUGE-L therefore mostly measures
"did the system stop", not "is the content right": a verbose-but-correct system
is punished by the recall-normalised F-measure denominator while a terse system
is rewarded.

This tool re-scores the *same* generations under three protocols so the
length confound can be reported explicitly instead of silently.

PROTOCOLS
---------
raw
    Prediction exactly as dumped (only outer whitespace stripped).  This is
    what eval_cvlm.py reported.

trunc_char
    Prediction truncated to `len(reference)` characters.  Oracle-length
    truncation: it hands every system the same character budget, so ROUGE-L
    precision/recall are computed over comparable spans.  It is an *oracle*
    (reference length is not known at inference time) and must be labelled as
    a diagnostic, never as a headline number.

trunc_word
    Same idea on whitespace words: first `len(reference.split())` words.
    Less sensitive to mid-word cuts than trunc_char.

first_sentence
    The first sentence of the prediction.  This is the only *non-oracle*
    length control here (no reference information is used), so it is the
    protocol that can honestly be reported as a main result alongside raw.

FIRST-SENTENCE HEURISTIC (deliberately simple and auditable)
------------------------------------------------------------
Scan for a sentence-final character in `.`, `!`, `?` (optionally followed by
closing quotes/brackets) that is followed by end-of-string or whitespace.  A
candidate boundary is REJECTED when:
  * the token ending at that `.` is a known abbreviation
    (Mr, Mrs, Ms, Dr, Prof, Sr, Jr, St, Mt, Gen, Col, Lt, Sgt, Rev, Hon,
    Inc, Ltd, Co, Corp, Dept, Univ, Fig, No, Vol, vs, etc, al, approx,
    e.g, i.e, a.m, p.m, U.S, U.K, U.N);
  * the character before the `.` is a single capital letter preceded by a
    non-letter (initials: "J. Smith", "H. G. Wells");
  * the `.` sits between two digits ("3.14") or is followed directly by a
    digit;
  * the next non-space character is lowercase (mid-sentence artefact).
A newline is also accepted as a sentence boundary (models often emit
list-style answers).  If no boundary is found the whole prediction is used.
The heuristic is intentionally not a full sentence splitter; it is applied
identically to every system, so any residual error is common-mode.

METRICS (per protocol)
----------------------
* ROUGE-L f-measure / precision / recall, `rouge_score` with Porter stemmer,
  averaged over samples (same estimator as eval_cvlm.py).
* BLEU-4, sacrebleu corpus-level (not sentence-averaged).
* mean prediction / reference length in characters and whitespace words.
* fraction of empty predictions (after strip).

CLUSTER (DOCUMENT-LEVEL) BOOTSTRAP
----------------------------------
The PwC test split has ~15 questions per source document and the file is
sorted in document blocks, so rows are *not* independent.  Row-level bootstrap
underestimates the variance.  This tool resamples DOCUMENTS with replacement
(all rows of a sampled document enter together, so per-document block size is
preserved) and recomputes the metric on the resample.  BLEU is resampled at
the level of sacrebleu's per-segment sufficient statistics, so the bootstrap
BLEU is a true corpus BLEU of the resampled corpus, not an average of
sentence BLEUs.

Document ids come from `--doc_ids FILE` (a JSON list of ids, position i = idx
i of the evaluated slice).  Build it once with `--make_doc_ids`:

    python tools/length_controlled_gen.py --make_doc_ids docids_2000.json \\
        --dataset sggetao/PwC --split "test[:2000]"

`test[:1000]` is a prefix of `test[:2000]`, so a single 2000-long id file
serves both the 2000-sample and the 1000-sample dumps.  Alignment holds
because the length filter in `cvlm_dataset.py` kept 100% of rows for these
runs (`kept 2000/2000`, `kept 1000/1000` in the job logs) and the eval
DataLoader runs with `shuffle=False`.

USAGE
-----
    python tools/length_controlled_gen.py \\
        --dumps runs/decisive16_fixed/*_dump.jsonl runs/rq3_grid/*_dump.jsonl \\
        --doc_ids docids_2000.json \\
        --compare vit_cr16:sfttrunc_first_cr16 \\
        --n_boot 2000 --out results.json --out_md table.md
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROTOCOLS = ("raw", "trunc_char", "trunc_word", "first_sentence")

# --------------------------------------------------------------------------- #
# first-sentence heuristic
# --------------------------------------------------------------------------- #

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "gen", "col",
    "lt", "sgt", "rev", "hon", "inc", "ltd", "co", "corp", "dept", "univ",
    "fig", "no", "vol", "vs", "etc", "al", "approx", "e.g", "i.e", "a.m",
    "p.m", "u.s", "u.k", "u.n", "cf", "ca", "op", "pp", "ed", "eds",
}
# token immediately preceding a candidate '.' boundary
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.']+$")
_CLOSERS = "\"')]}»”’"


def first_sentence(text: str) -> str:
    """Return the first sentence of ``text`` (see module docstring for rules)."""
    s = text.strip()
    if not s:
        return ""
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if ch == "\n":
            # a newline ends the "sentence" only if something was said already
            head = s[:i].strip()
            if head:
                return head
            i += 1
            continue
        if ch in ".!?":
            j = i + 1
            while j < n and s[j] in _CLOSERS:
                j += 1
            # boundary must be end-of-string or followed by whitespace
            if j < n and not s[j].isspace():
                i += 1
                continue
            if ch == ".":
                # digit context: 3.14 / v1.2
                if i > 0 and s[i - 1].isdigit() and j < n:
                    k = j
                    while k < n and s[k].isspace():
                        k += 1
                    if k < n and s[k].isdigit():
                        i += 1
                        continue
                # abbreviation
                m = _TOKEN_RE.search(s[:i])
                tok = (m.group(0) if m else "").rstrip(".").lower()
                if tok in _ABBREV:
                    i += 1
                    continue
                # single-letter initial: "J. Smith"
                if i >= 1 and s[i - 1].isalpha() and s[i - 1].isupper():
                    if i == 1 or not s[i - 2].isalpha():
                        i += 1
                        continue
            # next non-space char must not be lowercase
            k = j
            while k < n and s[k].isspace():
                k += 1
            if k < n and s[k].isalpha() and s[k].islower():
                i += 1
                continue
            return s[:j].strip()
        i += 1
    return s


# --------------------------------------------------------------------------- #
# protocols
# --------------------------------------------------------------------------- #


def apply_protocol(pred: str, ref: str, protocol: str) -> str:
    p = pred.strip()
    r = ref.strip()
    if protocol == "raw":
        return p
    if protocol == "trunc_char":
        return p[: len(r)]
    if protocol == "trunc_word":
        return " ".join(p.split()[: len(r.split())])
    if protocol == "first_sentence":
        return first_sentence(p)
    raise ValueError(f"unknown protocol {protocol!r}")


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


class Scorer:
    """Per-sample ROUGE-L plus resamplable sacrebleu sufficient statistics."""

    def __init__(self) -> None:
        from rouge_score import rouge_scorer
        from sacrebleu.metrics import BLEU

        self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self._bleu = BLEU()

    def rouge_per_sample(
        self, preds: Sequence[str], refs: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        f, p, r = [], [], []
        for hyp, ref in zip(preds, refs):
            sc = self._rouge.score(ref, hyp)["rougeL"]
            f.append(sc.fmeasure)
            p.append(sc.precision)
            r.append(sc.recall)
        return np.asarray(f), np.asarray(p), np.asarray(r)

    def bleu_stats(self, preds: Sequence[str], refs: Sequence[str]) -> np.ndarray:
        """Per-segment BLEU sufficient statistics, shape (n, 10)."""
        stats = self._bleu._extract_corpus_statistics(list(preds), [list(refs)])
        return np.asarray(stats, dtype=np.float64)

    def bleu_from_stats(self, stats_sum: np.ndarray) -> float:
        return float(self._bleu._compute_score_from_stats(list(stats_sum)).score)


def summarize(
    scorer: Scorer,
    preds: Sequence[str],
    refs: Sequence[str],
) -> Dict[str, object]:
    f, p, r = scorer.rouge_per_sample(preds, refs)
    stats = scorer.bleu_stats(preds, refs)
    pred_chars = np.asarray([len(x) for x in preds], dtype=np.float64)
    ref_chars = np.asarray([len(x) for x in refs], dtype=np.float64)
    pred_words = np.asarray([len(x.split()) for x in preds], dtype=np.float64)
    ref_words = np.asarray([len(x.split()) for x in refs], dtype=np.float64)
    empty = np.asarray([1.0 if not x.strip() else 0.0 for x in preds])
    return {
        "n": len(preds),
        "rougeL_f": float(f.mean()) if len(f) else 0.0,
        "rougeL_p": float(p.mean()) if len(p) else 0.0,
        "rougeL_r": float(r.mean()) if len(r) else 0.0,
        "bleu4": scorer.bleu_from_stats(stats.sum(axis=0)),
        "pred_chars": float(pred_chars.mean()) if len(preds) else 0.0,
        "ref_chars": float(ref_chars.mean()) if len(refs) else 0.0,
        "pred_words": float(pred_words.mean()) if len(preds) else 0.0,
        "ref_words": float(ref_words.mean()) if len(refs) else 0.0,
        "empty_frac": float(empty.mean()) if len(preds) else 0.0,
        "_per_sample": {"f": f, "p": p, "r": r, "bleu_stats": stats},
    }


# --------------------------------------------------------------------------- #
# cluster bootstrap
# --------------------------------------------------------------------------- #


def make_clusters(doc_ids: Sequence[str]) -> List[np.ndarray]:
    buckets: Dict[str, List[int]] = {}
    for i, d in enumerate(doc_ids):
        buckets.setdefault(d, []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in buckets.values()]


def bootstrap_indices(
    clusters: Sequence[np.ndarray], n_boot: int, seed: int
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_cl = len(clusters)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_cl, size=n_cl)
        out.append(np.concatenate([clusters[k] for k in pick]))
    return out


def ci_from_samples(vals: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi


# --------------------------------------------------------------------------- #
# dump loading
# --------------------------------------------------------------------------- #


def load_dump(path: str) -> Tuple[List[int], List[str], List[str], int]:
    idxs, preds, refs = [], [], []
    n_lines = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            rec = json.loads(line)
            if "prediction" not in rec or "reference" not in rec:
                continue
            idxs.append(int(rec["idx"]))
            preds.append(str(rec["prediction"]))
            refs.append(str(rec["reference"]))
    return idxs, preds, refs, n_lines


# --------------------------------------------------------------------------- #
# doc ids
# --------------------------------------------------------------------------- #


def make_doc_ids(dataset: str, split: str, field: str, out_path: str) -> None:
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    ids = [hashlib.md5(str(x).encode("utf-8")).hexdigest() for x in ds[field]]
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(ids, fh)
    uniq = len(set(ids))
    print(
        f"wrote {len(ids)} doc ids to {out_path}: {uniq} unique documents, "
        f"mean {len(ids)/max(uniq,1):.2f} questions/doc"
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", nargs="*", default=[], help="per-sample *.jsonl dumps (globs ok)")
    ap.add_argument("--labels", nargs="*", default=None, help="labels, one per dump (default: file stem)")
    ap.add_argument("--doc_ids", default=None, help="JSON list of doc ids, position i == dump idx i")
    ap.add_argument("--protocols", nargs="*", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--compare", nargs="*", default=[], help="paired bootstrap, 'LABEL_A:LABEL_B'")
    ap.add_argument("--compare_metrics", nargs="*", default=["rougeL_f", "bleu4"])
    ap.add_argument("--ci_metrics", nargs="*", default=["rougeL_f", "bleu4"])
    ap.add_argument("--out", default=None, help="write full results JSON here")
    ap.add_argument("--out_md", default=None, help="write markdown tables here")
    ap.add_argument("--make_doc_ids", default=None, help="build a doc-id file and exit")
    ap.add_argument("--dataset", default="sggetao/PwC")
    ap.add_argument("--split", default="test[:2000]")
    ap.add_argument("--doc_field", default="input")
    args = ap.parse_args()

    if args.make_doc_ids:
        make_doc_ids(args.dataset, args.split, args.doc_field, args.make_doc_ids)
        return 0

    paths: List[str] = []
    for pat in args.dumps:
        hits = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        paths.extend(hits)
    if not paths:
        ap.error("no dumps matched")
    if args.labels and len(args.labels) != len(paths):
        ap.error("--labels must have one entry per dump")

    doc_ids_all: Optional[List[str]] = None
    if args.doc_ids:
        with open(args.doc_ids, "r", encoding="utf-8") as fh:
            doc_ids_all = json.load(fh)

    scorer = Scorer()
    results: Dict[str, dict] = {}
    per_sample: Dict[str, dict] = {}
    skipped: List[Tuple[str, str]] = []

    for k, path in enumerate(paths):
        label = args.labels[k] if args.labels else os.path.basename(path).replace("_dump.jsonl", "").replace(".jsonl", "")
        idxs, preds, refs, n_lines = load_dump(path)
        if not preds:
            skipped.append((label, f"no prediction/reference fields ({n_lines} lines)"))
            print(f"[skip] {label}: dump has no generation fields ({n_lines} lines)", file=sys.stderr)
            continue
        entry: Dict[str, object] = {
            "path": os.path.abspath(path),
            "n_lines": n_lines,
            "n_with_generation": len(preds),
            "idx_min": min(idxs),
            "idx_max": max(idxs),
            "protocols": {},
        }
        per_sample[label] = {"idxs": np.asarray(idxs, dtype=np.int64), "protocols": {}}
        for proto in args.protocols:
            p2 = [apply_protocol(p, r, proto) for p, r in zip(preds, refs)]
            r2 = [r.strip() for r in refs]
            summ = summarize(scorer, p2, r2)
            ps = summ.pop("_per_sample")
            per_sample[label]["protocols"][proto] = ps
            entry["protocols"][proto] = summ  # type: ignore[index]
        # empty rate does not depend on the protocol for raw; keep raw's value
        entry["empty_frac"] = entry["protocols"]["raw"]["empty_frac"] if "raw" in entry["protocols"] else None  # type: ignore[index]
        results[label] = entry
        print(f"[ok] {label}: {len(preds)} generations (idx {min(idxs)}..{max(idxs)})", file=sys.stderr)

    # ---- bootstrap CIs -------------------------------------------------- #
    boot_cache: Dict[int, List[np.ndarray]] = {}

    def boot_for(n: int) -> Optional[List[np.ndarray]]:
        if doc_ids_all is None:
            return None
        if n not in boot_cache:
            ids = doc_ids_all[:n]
            clusters = make_clusters(ids)
            boot_cache[n] = bootstrap_indices(clusters, args.n_boot, args.seed)
        return boot_cache[n]

    def metric_on(label: str, proto: str, metric: str, sel: np.ndarray) -> float:
        ps = per_sample[label]["protocols"][proto]
        if metric == "rougeL_f":
            return float(ps["f"][sel].mean())
        if metric == "rougeL_p":
            return float(ps["p"][sel].mean())
        if metric == "rougeL_r":
            return float(ps["r"][sel].mean())
        if metric == "bleu4":
            return scorer.bleu_from_stats(ps["bleu_stats"][sel].sum(axis=0))
        raise ValueError(metric)

    if doc_ids_all is not None:
        for label, entry in results.items():
            n = int(entry["n_with_generation"])  # type: ignore[arg-type]
            boots = boot_for(n)
            assert boots is not None
            for proto in args.protocols:
                for metric in args.ci_metrics:
                    vals = np.asarray([metric_on(label, proto, metric, b) for b in boots])
                    lo, hi = ci_from_samples(vals, args.alpha)
                    entry["protocols"][proto][f"{metric}_ci"] = [lo, hi]  # type: ignore[index]
            print(f"[ci] {label} done", file=sys.stderr)
        n_docs = {n: len(make_clusters(doc_ids_all[:n])) for n in boot_cache}
        results["_bootstrap"] = {
            "kind": "cluster (document-level) percentile bootstrap",
            "n_boot": args.n_boot,
            "seed": args.seed,
            "alpha": args.alpha,
            "n_documents": n_docs,
        }
    else:
        print("[warn] --doc_ids not given: no CIs computed", file=sys.stderr)

    # ---- paired comparisons --------------------------------------------- #
    comparisons = []
    for spec in args.compare:
        a, b = spec.split(":", 1)
        if a not in per_sample or b not in per_sample:
            print(f"[warn] compare {spec}: missing label", file=sys.stderr)
            continue
        ia, ib = per_sample[a]["idxs"], per_sample[b]["idxs"]
        common = np.intersect1d(ia, ib)
        pos_a = {int(v): i for i, v in enumerate(ia)}
        pos_b = {int(v): i for i, v in enumerate(ib)}
        sel_a = np.asarray([pos_a[int(v)] for v in common])
        sel_b = np.asarray([pos_b[int(v)] for v in common])
        if doc_ids_all is None:
            print(f"[warn] compare {spec}: no --doc_ids, skipping", file=sys.stderr)
            continue
        clusters = make_clusters([doc_ids_all[int(v)] for v in common])
        boots = bootstrap_indices(clusters, args.n_boot, args.seed)
        for proto in args.protocols:
            for metric in args.compare_metrics:
                pa = per_sample[a]["protocols"][proto]
                pb = per_sample[b]["protocols"][proto]

                def val(ps, sel):
                    if metric == "bleu4":
                        return scorer.bleu_from_stats(ps["bleu_stats"][sel].sum(axis=0))
                    key = {"rougeL_f": "f", "rougeL_p": "p", "rougeL_r": "r"}[metric]
                    return float(ps[key][sel].mean())

                point = val(pa, sel_a) - val(pb, sel_b)
                deltas = np.asarray([val(pa, sel_a[bi]) - val(pb, sel_b[bi]) for bi in boots])
                lo, hi = ci_from_samples(deltas, args.alpha)
                comparisons.append(
                    {
                        "a": a,
                        "b": b,
                        "protocol": proto,
                        "metric": metric,
                        "a_value": val(pa, sel_a),
                        "b_value": val(pb, sel_b),
                        "delta": point,
                        "ci": [lo, hi],
                        "p_a_gt_b": float((deltas > 0).mean()),
                        "n_common": int(len(common)),
                        "n_documents": len(clusters),
                    }
                )
        print(f"[cmp] {a} vs {b} done", file=sys.stderr)

    payload = {"configs": results, "comparisons": comparisons, "skipped": skipped}

    # ---- markdown -------------------------------------------------------- #
    lines: List[str] = []
    lines.append("## Generative metrics by length protocol\n")
    hdr = (
        "| config | protocol | ROUGE-L F | ROUGE-L P | ROUGE-L R | BLEU-4 | "
        "pred chars | ref chars | pred words | ref words | empty % |"
    )
    lines.append(hdr)
    lines.append("|" + "---|" * 11)
    for label, entry in results.items():
        if label.startswith("_"):
            continue
        for proto in args.protocols:
            s = entry["protocols"][proto]  # type: ignore[index]
            ci_f = s.get("rougeL_f_ci")
            ci_b = s.get("bleu4_ci")
            f_txt = f"{s['rougeL_f']:.4f}" + (f" [{ci_f[0]:.4f}, {ci_f[1]:.4f}]" if ci_f else "")
            b_txt = f"{s['bleu4']:.2f}" + (f" [{ci_b[0]:.2f}, {ci_b[1]:.2f}]" if ci_b else "")
            lines.append(
                f"| {label} | {proto} | {f_txt} | {s['rougeL_p']:.4f} | {s['rougeL_r']:.4f} | "
                f"{b_txt} | {s['pred_chars']:.1f} | {s['ref_chars']:.1f} | "
                f"{s['pred_words']:.1f} | {s['ref_words']:.1f} | {100*s['empty_frac']:.1f} |"
            )
    if comparisons:
        lines.append("\n## Paired cluster-bootstrap comparisons\n")
        lines.append("| A | B | protocol | metric | A | B | delta (A-B) | 95% CI | P(A>B) | n | docs |")
        lines.append("|" + "---|" * 11)
        for c in comparisons:
            fmt = "{:.4f}" if c["metric"].startswith("rouge") else "{:.2f}"
            lines.append(
                f"| {c['a']} | {c['b']} | {c['protocol']} | {c['metric']} | "
                f"{fmt.format(c['a_value'])} | {fmt.format(c['b_value'])} | {fmt.format(c['delta'])} | "
                f"[{fmt.format(c['ci'][0])}, {fmt.format(c['ci'][1])}] | {c['p_a_gt_b']:.3f} | "
                f"{c['n_common']} | {c['n_documents']} |"
            )
    if skipped:
        lines.append("\n## Skipped dumps\n")
        for label, why in skipped:
            lines.append(f"* `{label}`: {why}")
    md = "\n".join(lines)
    print(md)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
