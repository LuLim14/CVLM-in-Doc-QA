#!/usr/bin/env python3
"""LLMLingua-2 as an external published prompt-compression baseline.

Why this tool exists
--------------------
CVLM compresses a document into ``N_i = min(ceil(L_enc_i / cr), max_vision_len)``
decoder slots. The truncation baselines (``--doc_truncation first/last/random/
oracle``) spend exactly the same budget, and so does the skim-then-read
selector (``tools/score_chunks.py``). LLMLingua-2 (Pan et al., ACL Findings
2024, ``microsoft/llmlingua-2-xlm-roberta-large-meetingbank``) is the obvious
*published* text-space competitor, but its native knob is a compression *rate*
counted in its own (XLM-R / cl100k) token space. Comparing it at "rate 1/8"
would silently give it a different number of decoder slots than CVLM, and the
whole equal-budget claim would collapse.

This tool therefore runs LLMLingua-2 offline over the documents of a split and
emits the SAME JSONL contract ``tools/score_chunks.py`` emits, so the eval
consumes it with the existing, untouched machinery::

    python src/eval_cvlm.py --mode baseline_llm_full \
        --doc_truncation file --doc_selection_file <this output>

Protocol (literal)
------------------
1. ONLY the document (dataset field ``input``) is compressed. The question
   (field ``prompt``) is never passed to the compressor and is never
   compressed — ``CvlmTrainDataset`` appends ``"\\n\\n" + question`` to the
   selected text itself, exactly as for every other doc_truncation strategy.
   The selection is therefore query-INDEPENDENT by construction; PwC repeats
   one document across ~15 questions and ``tools/audit_selection_jsonl.py``
   verifies that all of them received a byte-identical ``selected_text``.
2. The budget is the CVLM latent count, measured in **SmolLM2 decoder
   tokens**: ``V_i = max(min(ceil(min(L_ModernBERT_i, max_source_len) / cr),
   max_vision_len), 1)`` — byte for byte the formula in
   ``CvlmTrainDataset._truncated_doc_ids`` (``--doc_budget_cr``).
   LLMLingua-2's own rate is calibrated against that budget (see below); the
   result is then hard-truncated to AT MOST ``V_i`` decoder tokens. Every
   written line is re-tokenised and asserted at write time: overshoot is a
   fatal bug (``--doc_truncation=file`` trusts the JSONL blindly and would
   silently hand the baseline extra slots); shortfall is kept as-is and
   reported, since it can only hurt this baseline, never inflate it.
3. Rate calibration: LLMLingua-2's ``rate`` is a fraction of retained words in
   ITS tokenizer's space, which is not the decoder's space. We start from
   ``rate = overshoot * V_i / L_dec_i`` (``L_dec_i`` = document length in
   decoder tokens) and, while the compressed text is still *below* budget,
   re-run with the rate rescaled by the observed miss, up to
   ``--calibrate_rounds`` extra calls. Every round is logged per sample
   (``n_calls``, ``rate``), so the cost of hitting the budget is auditable.

``--force_tokens`` defaults to EMPTY on purpose. The LLMLingua-2 demo uses
``force_tokens=['\\n', '?']``, which is sane at rate 0.33; at the rates this
protocol demands (~0.12 at cr=8, ~0.06 at cr=16) forcing every newline would
spend a third to three quarters of the whole budget on newline tokens — the
PwC documents used here average ~25 newlines against a ~65 token budget at
cr=8. Keeping punctuation would be a self-inflicted handicap on the baseline,
not a faithful reproduction. Pass ``--force_tokens '\\n,?'`` to restore the
demo behaviour.

Documents repeat across rows (PwC asks several questions about one document),
and the compression is question-independent, so identical documents are
compressed once and cached (``--no_cache`` disables). The stats report both
wall time and the mean per-call latency, so the un-cached cost is recoverable.

Output JSONL (one line per sample of the filtered dataset, idx = position in
that filtered dataset — the contract ``CvlmTrainDataset._load_doc_selection``
validates). The loader only checks the line count and the idx range, which
would not catch a permuted or stale file, so every line also carries the
provenance ``tools/audit_selection_jsonl.py`` re-checks against the dataset::

    {"idx": int, "selected_text": str,
     "source_sha256": str, "prompt_sha256": str,
     "selected_smol_tokens": int, "target_budget": int, "cr": int,
     "llmlingua_model": str, "model_revision": str,
     "l_enc": int, "l_dec": int, "raw_smol_tokens": int,
     "rate": float, "n_calls": int, "truncated": bool,
     "compress_seconds": float}

Example::

    python tools/llmlingua_select.py \
        --dataset_split "test[2100:5100]" --max_samples 0 \
        --decoder_tokenizer_name HuggingFaceTB/SmolLM2-1.7B-Instruct \
        --encoder_tokenizer_name answerdotai/ModernBERT-base \
        --max_prompt_len 512 --max_answer_len 1024 --max_source_len 1024 \
        --budget_cr 8 --max_vision_len 1024 \
        --output_jsonl results/llmlingua_cr8.jsonl \
        --stats_json results/llmlingua_cr8_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Tuple

# Nothing here trains or computes a loss; liger's fused patches are pure risk.
os.environ.setdefault("DISABLE_LIGER", "1")

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLMLingua-2 document compression at the CVLM decoder-slot budget")
    # Dataset flags — MUST match the eval_cvlm.py run that consumes the JSONL:
    # idx alignment is by position in the FILTERED dataset.
    p.add_argument("--dataset_name", type=str, default="sggetao/PwC")
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument(
        "--decoder_tokenizer_name",
        type=str,
        default=os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
        help="Decoder tokenizer defining the budget unit. MUST be the tokenizer the eval "
        "decoder uses (the truncated-SFT dirs carry a byte-identical copy of the base one).",
    )
    p.add_argument(
        "--encoder_tokenizer_name",
        type=str,
        default=os.environ.get("TEXT_ENCODER_NAME", "answerdotai/ModernBERT-base"),
        help="Encoder tokenizer defining L_enc in the budget formula.",
    )
    p.add_argument("--max_prompt_len", type=int, default=512)
    p.add_argument("--max_answer_len", type=int, default=2048)
    p.add_argument("--max_source_len", type=int, default=1024)
    p.add_argument("--max_samples", type=int, default=0, help="Limit to first N samples (0 = all).")
    # Budget.
    p.add_argument(
        "--budget_cr",
        type=int,
        required=True,
        help="Decoder-token budget N = max(min(ceil(L_enc/N), max_vision_len), 1) — the CVLM "
        "vision-token budget at compression_rate=N; same formula as eval_cvlm.py --doc_budget_cr.",
    )
    p.add_argument("--max_vision_len", type=int, default=1024, help="Budget cap (doc_budget_cap).")
    # LLMLingua-2.
    p.add_argument(
        "--llmlingua_model",
        type=str,
        default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        help="Published LLMLingua-2 token-classification compressor.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--force_tokens",
        type=str,
        default="",
        help="Comma-separated tokens LLMLingua-2 must retain (demo default: '\\n,?'). Empty by "
        "default — see the module docstring: at these budgets forcing punctuation is a handicap.",
    )
    p.add_argument("--drop_consecutive", action="store_true", help="Drop repeated force_tokens.")
    p.add_argument(
        "--calibrate_rounds",
        type=int,
        default=2,
        help="Extra compress calls allowed to raise the rate when the first pass lands under "
        "budget (0 = single shot). Every call is counted in the per-sample dump.",
    )
    p.add_argument(
        "--overshoot",
        type=float,
        default=1.10,
        help="Rate safety factor: aim for overshoot*budget decoder tokens, then truncate down "
        "to exactly the budget. >1 trades a little truncation for less shortfall.",
    )
    p.add_argument("--no_cache", action="store_true", help="Do not reuse compressions of identical documents.")
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--stats_json", type=str, default="", help="Optional JSON with budget/timing statistics.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_revision(model_name: str) -> str:
    """Commit sha of the locally cached snapshot ('' when undeterminable).

    Recorded per line so a reviewer can tell which weights produced the
    selection without trusting the model name alone.
    """
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(model_name, local_files_only=True)
        return os.path.basename(os.path.normpath(path))
    except Exception:  # offline / non-hub path / hub API change
        return ""


def truncate_to_budget(dec_tok, text: str, budget: int) -> Tuple[str, int]:
    """Cut `text` to at most `budget` decoder tokens, re-tokenisation-stable.

    The consumer (``CvlmTrainDataset``) re-tokenises the raw string we write,
    so it is not enough to slice ids: the decoded prefix must itself tokenise
    to <= budget tokens. Byte-level BPE round-trips exactly in the common case;
    the loop only ever fires on the rare boundary where it does not.

    Returns (text, achieved_decoder_token_count).
    """

    def n_tok(s: str) -> int:
        return len(dec_tok(s, add_special_tokens=False, truncation=False)["input_ids"])

    ids = dec_tok(text, add_special_tokens=False, truncation=False)["input_ids"]
    if len(ids) <= budget:
        return text, len(ids)
    k = budget
    while k > 0:
        cand = dec_tok.decode(ids[:k], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        m = n_tok(cand)
        if m <= budget:
            return cand, m
        k -= 1
    return "", 0


def main() -> None:
    args = parse_args()
    if args.budget_cr <= 0:
        raise ValueError("--budget_cr must be > 0")
    if args.max_vision_len <= 0:
        raise ValueError("--max_vision_len must be > 0")
    if args.overshoot < 1.0:
        raise ValueError("--overshoot must be >= 1.0")

    import numpy as np
    import torch
    from tqdm import tqdm

    from cvlm_dataset import CvlmTrainDataset

    torch.manual_seed(args.seed)

    force_tokens: List[str] = [
        t.replace("\\n", "\n").replace("\\t", "\t") for t in args.force_tokens.split(",") if t != ""
    ]

    # Same class, same flags => byte-identical filter => idx alignment holds.
    print("Building the filtered dataset (tokenizers only, no model)...")
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
    n_total = len(dataset)

    print(f"Loading LLMLingua-2 ({args.llmlingua_model}) on {args.device}...")
    from llmlingua import PromptCompressor

    t_load = time.time()
    compressor = PromptCompressor(
        model_name=args.llmlingua_model,
        use_llmlingua2=True,
        device_map=args.device,
    )
    print(f"  loaded in {time.time() - t_load:.1f}s")

    def n_dec(s: str) -> int:
        return len(dec_tok(s, add_special_tokens=False, truncation=False)["input_ids"])

    call_times: List[float] = []

    def compress_one(document: str, budget: int) -> Dict[str, object]:
        """Compress one document down to <= `budget` decoder tokens."""
        l_dec = n_dec(document)
        if l_dec <= budget:
            # Already inside the budget: compressing would only destroy
            # information the budget can afford. Same convention as the
            # truncation strategies (a short doc is kept whole).
            return {
                "selected_text": document,
                "dec_tokens": l_dec,
                "raw_dec_tokens": l_dec,
                "l_dec": l_dec,
                "rate": 1.0,
                "n_calls": 0,
                "truncated": False,
                "compress_seconds": 0.0,
            }
        rate = min(1.0, args.overshoot * budget / max(l_dec, 1))
        text = ""
        raw = 0
        n_calls = 0
        spent = 0.0
        for _ in range(max(args.calibrate_rounds, 0) + 1):
            t0 = time.time()
            res = compressor.compress_prompt_llmlingua2(
                document,
                rate=rate,
                force_tokens=force_tokens,
                drop_consecutive=args.drop_consecutive,
            )
            dt = time.time() - t0
            call_times.append(dt)
            spent += dt
            n_calls += 1
            text = str(res["compressed_prompt"])
            raw = n_dec(text)
            if raw >= budget or rate >= 1.0:
                break
            # Under budget: scale the rate by the observed miss and retry.
            rate = min(1.0, rate * (args.overshoot * budget) / max(raw, 1))
        final_text, achieved = truncate_to_budget(dec_tok, text, budget)
        return {
            "selected_text": final_text,
            "dec_tokens": achieved,
            "raw_dec_tokens": raw,
            "l_dec": l_dec,
            "rate": float(rate),
            "n_calls": n_calls,
            "truncated": bool(raw > achieved),
            "compress_seconds": float(spent),
        }

    revision = resolve_revision(args.llmlingua_model)
    print(f"  llmlingua-2 snapshot revision: {revision or '<unknown>'}")

    cache: Dict[str, Dict[str, object]] = {}
    budgets: List[int] = []
    achieved_all: List[int] = []
    n_cache_hits = 0
    n_empty = 0

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    t_start = time.time()
    with open(args.output_jsonl, "w") as fh, torch.no_grad():
        for idx in tqdm(range(n_total), desc=f"llmlingua-2 (budget_cr={args.budget_cr})"):
            record = dataset._hf[dataset._row_indices[idx]]
            document = str(record["input"])
            question = str(record["prompt"])
            # Budget: identical measurement to CvlmTrainDataset._truncated_doc_ids.
            l_enc = min(
                len(enc_tok(document, add_special_tokens=False, truncation=False)["input_ids"]),
                dataset.max_source_len,
            )
            budget = max(min((l_enc + args.budget_cr - 1) // args.budget_cr, args.max_vision_len), 1)

            key = f"{budget}\x00{document}"
            out = None if args.no_cache else cache.get(key)
            if out is None:
                out = compress_one(document, budget)
                if not args.no_cache:
                    cache[key] = out
            else:
                n_cache_hits += 1

            selected_text = str(out["selected_text"])
            if not selected_text.strip():
                n_empty += 1
            # Runtime assert on EVERY written line. eval_cvlm.py --doc_truncation=file
            # trusts this JSONL without re-measuring, so a silent overshoot here would
            # hand the baseline decoder slots CVLM never had.
            achieved = n_dec(selected_text)
            if achieved != int(out["dec_tokens"]):
                raise RuntimeError(
                    f"idx={idx}: re-tokenisation is unstable "
                    f"({achieved} != {out['dec_tokens']} decoder tokens)"
                )
            if achieved > budget:
                raise RuntimeError(
                    f"idx={idx}: selection exceeds the decoder-slot budget "
                    f"({achieved} > V_i={budget}); refusing to write an over-budget baseline"
                )
            rec = {
                "idx": idx,
                "selected_text": selected_text,
                "source_sha256": sha256_text(document),
                "prompt_sha256": sha256_text(question),
                "selected_smol_tokens": int(achieved),
                "target_budget": int(budget),
                "cr": int(args.budget_cr),
                "llmlingua_model": args.llmlingua_model,
                "model_revision": revision,
                "l_enc": int(l_enc),
                "l_dec": int(out["l_dec"]),
                "raw_smol_tokens": int(out["raw_dec_tokens"]),
                "rate": float(out["rate"]),
                "n_calls": int(out["n_calls"]),
                "truncated": bool(out["truncated"]),
                "compress_seconds": float(out["compress_seconds"]),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            budgets.append(int(budget))
            achieved_all.append(int(achieved))

    wall = time.time() - t_start
    b = np.asarray(budgets, dtype=np.float64)
    a = np.asarray(achieved_all, dtype=np.float64)
    deficit = b - a
    under = deficit > 0
    calls = np.asarray(call_times, dtype=np.float64)
    stats = {
        "output_jsonl": args.output_jsonl,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "budget_cr": int(args.budget_cr),
        "max_vision_len": int(args.max_vision_len),
        "llmlingua_model": args.llmlingua_model,
        "model_revision": revision,
        "force_tokens": force_tokens,
        "calibrate_rounds": int(args.calibrate_rounds),
        "overshoot": float(args.overshoot),
        "n_samples": int(n_total),
        "n_unique_documents": int(len(cache)) if not args.no_cache else -1,
        "n_cache_hits": int(n_cache_hits),
        "budget_mean": float(b.mean()),
        "budget_median": float(np.median(b)),
        "budget_min": int(b.min()),
        "budget_max": int(b.max()),
        "achieved_mean": float(a.mean()),
        "achieved_median": float(np.median(a)),
        "achieved_min": int(a.min()),
        "achieved_max": int(a.max()),
        "frac_exact_budget": float((deficit == 0).mean()),
        "frac_under_budget": float(under.mean()),
        "mean_deficit_all": float(deficit.mean()),
        "mean_deficit_when_under": float(deficit[under].mean()) if under.any() else 0.0,
        "max_deficit": int(deficit.max()),
        "frac_over_budget": float((a > b).mean()),  # must be 0.0
        "n_empty_selections": int(n_empty),
        "n_compress_calls": int(calls.size),
        "calls_per_unique_document": float(calls.size / max(len(cache), 1)) if cache else float("nan"),
        "compress_call_seconds_mean": float(calls.mean()) if calls.size else 0.0,
        "compress_seconds_total": float(calls.sum()),
        # Cost of the LLMLingua-2 forward passes alone, per unique document.
        "compress_seconds_per_document": float(calls.sum() / max(len(cache), 1)) if cache else 0.0,
        "wall_seconds_total": float(wall),
        "wall_seconds_per_sample": float(wall / max(n_total, 1)),
        # What the same 3000 rows would have cost with the document cache OFF
        # (PwC repeats each document across ~15 questions).
        "estimated_seconds_without_cache": float(calls.sum() / max(len(cache), 1) * n_total)
        if cache
        else float(calls.sum()),
    }
    if float((a > b).mean()) != 0.0:
        raise RuntimeError("BUG: some selections exceed the decoder-token budget")
    print(f"\nWrote {n_total} lines to {args.output_jsonl}")
    for k, v in stats.items():
        print(f"  {k:34s}: {v}")
    if args.stats_json:
        os.makedirs(os.path.dirname(args.stats_json) or ".", exist_ok=True)
        with open(args.stats_json, "w") as fh:
            json.dump(stats, fh, indent=2)
        print(f"Stats -> {args.stats_json}")
    print(
        "Consume with: python src/eval_cvlm.py --mode baseline_llm_full|sft "
        f"--doc_truncation file --doc_selection_file {args.output_jsonl} "
        "(same dataset/filter flags!)"
    )


if __name__ == "__main__":
    main()
