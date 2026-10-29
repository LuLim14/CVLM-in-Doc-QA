#!/usr/bin/env python3
"""Preflight audit of a --doc_selection_file JSONL. CPU only, no GPU needed.

Why this exists
---------------
``CvlmTrainDataset._load_doc_selection`` validates exactly two things: the
number of lines equals the number of filtered samples, and every ``idx`` is in
range and unique. That is enough to catch a wrong split; it is NOT enough to
catch a *permuted*, stale or partially regenerated file — the documents and the
questions would be silently mispaired and the eval would still produce
plausible-looking numbers. It also never re-measures the token budget: the
"equal budget" claim rests entirely on whoever wrote the JSONL.

This tool closes both gaps before any GPU time is spent:

  * line count / idx coverage (the loader's own checks, run early);
  * ``source_sha256`` and ``prompt_sha256`` of every line match the document
    and the question the eval will actually pair with that idx;
  * ``target_budget`` equals the independently recomputed CVLM latent count
    ``V_i = max(min(ceil(min(L_ModernBERT_i, max_source_len)/cr), max_vision_len), 1)``;
  * ``selected_smol_tokens`` equals the decoder-tokenised length of
    ``selected_text`` measured HERE, and never exceeds ``V_i`` (fatal);
    shortfall is reported, not fatal (the protocol keeps short selections);
  * query invariance: PwC repeats one document across ~15 questions, and a
    document-only compressor must return byte-identical ``selected_text`` for
    all of them. A violation is FATAL: the baseline would silently be
    query-aware, which is a different (and stronger) method than the one the
    write-up claims. ``--allow_query_aware`` downgrades it to a warning for
    anyone deliberately building a query-conditioned variant.

Exit code is non-zero on any fatal finding, so ``set -e`` in the sbatch stops
before the evals run.

Example::

    python tools/audit_selection_jsonl.py \
        --jsonl results/llmlingua_cr8.jsonl --budget_cr 8 \
        --dataset_split "test[2100:5100]" --max_source_len 1024 \
        --decoder_tokenizer_name HuggingFaceTB/SmolLM2-1.7B-Instruct \
        --out_json results/llmlingua_cr8_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

os.environ.setdefault("DISABLE_LIGER", "1")
# Tokenizers and hashing only — nothing is ever moved to a device, so this runs
# on a login node just as well as inside the job.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit a doc_selection_file JSONL against the dataset")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--budget_cr", type=int, required=True)
    p.add_argument("--max_vision_len", type=int, default=1024)
    p.add_argument("--dataset_name", type=str, default="sggetao/PwC")
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument(
        "--decoder_tokenizer_name",
        type=str,
        default=os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    )
    p.add_argument(
        "--encoder_tokenizer_name",
        type=str,
        default=os.environ.get("TEXT_ENCODER_NAME", "answerdotai/ModernBERT-base"),
    )
    p.add_argument("--max_prompt_len", type=int, default=512)
    p.add_argument("--max_answer_len", type=int, default=2048)
    p.add_argument("--max_source_len", type=int, default=1024)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument(
        "--allow_query_aware",
        action="store_true",
        help="Downgrade 'one document got different selections for different questions' from "
        "fatal to a warning. Only for a deliberately query-conditioned variant.",
    )
    p.add_argument("--expect_rows", type=int, default=0, help="Fatal unless the file has exactly N rows (0 = off).")
    p.add_argument(
        "--expect_docs", type=int, default=0, help="Fatal unless there are exactly N unique documents (0 = off)."
    )
    p.add_argument("--out_json", type=str, default="")
    return p.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    from cvlm_dataset import CvlmTrainDataset

    print(f"Auditing {args.jsonl} (cr={args.budget_cr})")
    dataset = CvlmTrainDataset(
        hf_dataset_name=args.dataset_name,
        hf_split=args.dataset_split,
        decoder_tokenizer_name=args.decoder_tokenizer_name,
        encoder_tokenizer_name=args.encoder_tokenizer_name,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
        max_source_len=args.max_source_len,
        max_samples=args.max_samples,
    )
    enc_tok = dataset._enc_tok
    dec_tok = dataset._dec_tok
    n = len(dataset)

    records: List[Dict] = []
    with open(args.jsonl) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"FATAL {args.jsonl}:{line_no}: invalid JSON: {e}")

    fatal: List[str] = []
    warn: List[str] = []
    if len(records) != n:
        fatal.append(f"line count {len(records)} != filtered dataset size {n}")
    seen = set()
    for rec in records:
        i = int(rec["idx"])
        if i in seen:
            fatal.append(f"duplicate idx {i}")
        if not 0 <= i < n:
            fatal.append(f"idx {i} out of range [0,{n})")
        seen.add(i)
    if len(seen) != n:
        fatal.append(f"idx coverage {len(seen)} != {n} (missing indices)")
    if fatal:
        for m in fatal:
            print(f"  FATAL: {m}")
        raise SystemExit(1)

    by_idx = {int(r["idx"]): r for r in records}
    by_doc: Dict[str, List[int]] = defaultdict(list)
    n_budget_mismatch = 0
    n_token_mismatch = 0
    n_over = 0
    n_under = 0
    deficit_sum = 0
    deficit_max = 0
    deficits: List[int] = []
    n_hash_doc_bad = 0
    n_hash_q_bad = 0
    n_calls_sum = 0
    compress_seconds_sum = 0.0
    n_truncated = 0
    models = set()
    revisions = set()
    search_modes = set()
    n_cr_mismatch = 0

    for i in range(n):
        rec = by_idx[i]
        row = dataset._hf[dataset._row_indices[i]]
        document = str(row["input"])
        question = str(row["prompt"])
        doc_sha = sha256_text(document)
        if str(rec.get("source_sha256", "")) != doc_sha:
            n_hash_doc_bad += 1
        if str(rec.get("prompt_sha256", "")) != sha256_text(question):
            n_hash_q_bad += 1

        l_enc = min(
            len(enc_tok(document, add_special_tokens=False, truncation=False)["input_ids"]),
            args.max_source_len,
        )
        v_i = max(min((l_enc + args.budget_cr - 1) // args.budget_cr, args.max_vision_len), 1)
        if int(rec.get("target_budget", -1)) != v_i:
            n_budget_mismatch += 1

        text = str(rec["selected_text"])
        measured = len(dec_tok(text, add_special_tokens=False, truncation=False)["input_ids"])
        if int(rec.get("selected_smol_tokens", -1)) != measured:
            n_token_mismatch += 1
        if measured > v_i:
            n_over += 1
        elif measured < v_i:
            n_under += 1
            deficit_sum += v_i - measured
            deficit_max = max(deficit_max, v_i - measured)
        deficits.append(v_i - measured)

        by_doc[doc_sha].append(i)
        n_calls_sum += int(rec.get("n_calls", 0))
        compress_seconds_sum += float(rec.get("compress_seconds", 0.0))
        n_truncated += int(bool(rec.get("truncated", False)))
        models.add(str(rec.get("llmlingua_model", "")))
        revisions.add(str(rec.get("model_revision", "")))
        search_modes.add(str(rec.get("search_mode", "")))
        if int(rec.get("cr", -1)) != args.budget_cr:
            n_cr_mismatch += 1

    # Query invariance: one document, many questions, one selection.
    n_docs_multi = 0
    n_docs_inconsistent = 0
    for doc_sha, idxs in by_doc.items():
        if len(idxs) > 1:
            n_docs_multi += 1
            texts = {by_idx[i]["selected_text"] for i in idxs}
            if len(texts) > 1:
                n_docs_inconsistent += 1

    if n_hash_doc_bad:
        fatal.append(f"{n_hash_doc_bad}/{n} source_sha256 mismatches (documents mispaired)")
    if n_hash_q_bad:
        fatal.append(f"{n_hash_q_bad}/{n} prompt_sha256 mismatches (questions mispaired)")
    if n_budget_mismatch:
        fatal.append(f"{n_budget_mismatch}/{n} target_budget != recomputed V_i")
    if n_token_mismatch:
        fatal.append(f"{n_token_mismatch}/{n} selected_smol_tokens != measured decoder tokens")
    if n_over:
        fatal.append(f"{n_over}/{n} selections EXCEED the budget V_i (unequal budget)")
    if n_truncated:
        fatal.append(
            f"{n_truncated}/{n} selections used prefix truncation; the main protocol "
            "requires whole LLMLingua outputs selected only through the threshold knob"
        )
    if len(models) > 1:
        fatal.append(f"mixed llmlingua_model values in one file: {sorted(models)}")
    if "" in models:
        fatal.append("missing llmlingua_model provenance")
    if len(revisions) > 1 or "" in revisions:
        fatal.append(f"missing or mixed model_revision provenance: {sorted(revisions)}")
    if len(search_modes) > 1 or "" in search_modes:
        fatal.append(f"missing or mixed search_mode provenance: {sorted(search_modes)}")
    if n_cr_mismatch:
        fatal.append(f"{n_cr_mismatch}/{n} records have cr != --budget_cr={args.budget_cr}")
    if n_docs_inconsistent:
        msg = (
            f"{n_docs_inconsistent}/{n_docs_multi} documents got different selections for "
            f"different questions -> the baseline is QUERY-AWARE, not the document-only "
            f"method claimed in the protocol"
        )
        (warn if args.allow_query_aware else fatal).append(msg)
    if args.expect_rows and n != args.expect_rows:
        fatal.append(f"row count {n} != expected {args.expect_rows}")
    if args.expect_docs and len(by_doc) != args.expect_docs:
        fatal.append(f"unique document count {len(by_doc)} != expected {args.expect_docs}")

    out = {
        "jsonl": args.jsonl,
        "dataset_split": args.dataset_split,
        "cr": int(args.budget_cr),
        "n_rows": n,
        "n_unique_documents": len(by_doc),
        "n_documents_with_multiple_questions": n_docs_multi,
        "n_documents_with_inconsistent_selection": n_docs_inconsistent,
        "query_invariant": n_docs_inconsistent == 0,
        "n_over_budget": n_over,
        "n_under_budget": n_under,
        "frac_under_budget": n_under / max(n, 1),
        "frac_exact_budget": (n - n_under - n_over) / max(n, 1),
        "mean_deficit_all": deficit_sum / max(n, 1),
        "median_deficit_all": float(np.median(deficits)) if deficits else 0.0,
        "p90_deficit_all": float(np.percentile(deficits, 90)) if deficits else 0.0,
        "p95_deficit_all": float(np.percentile(deficits, 95)) if deficits else 0.0,
        "mean_deficit_when_under": deficit_sum / n_under if n_under else 0.0,
        "max_deficit": deficit_max,
        "n_source_sha_mismatch": n_hash_doc_bad,
        "n_prompt_sha_mismatch": n_hash_q_bad,
        "n_budget_mismatch": n_budget_mismatch,
        "n_token_count_mismatch": n_token_mismatch,
        "n_prefix_truncated": n_truncated,
        "llmlingua_model": sorted(models),
        "model_revision": sorted(revisions),
        "search_mode": sorted(search_modes),
        "n_cr_mismatch": n_cr_mismatch,
        "compress_calls_total": n_calls_sum,
        "compress_seconds_total_from_jsonl": compress_seconds_sum,
        "fatal": fatal,
        "warnings": warn,
    }
    for k, v in out.items():
        if k not in ("fatal", "warnings"):
            print(f"  {k:44s}: {v}")
    for m in warn:
        print(f"  WARNING: {m}")
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"Audit -> {args.out_json}")
    if fatal:
        for m in fatal:
            print(f"  FATAL: {m}")
        raise SystemExit(1)
    print("AUDIT OK")


if __name__ == "__main__":
    main()
