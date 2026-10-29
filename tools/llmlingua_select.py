#!/usr/bin/env python3
"""LLMLingua-2 as an external published prompt-compression baseline.

Why this tool exists
--------------------
CVLM compresses a document into ``V_i = min(ceil(L_enc_i / cr), max_vision_len)``
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
   verifies (fatally) that all of them got a byte-identical ``selected_text``.
2. The budget is the CVLM latent count, measured in **SmolLM2 decoder
   tokens**: ``V_i = max(min(ceil(min(L_ModernBERT_i, max_source_len) / cr),
   max_vision_len), 1)`` — byte for byte the formula in
   ``CvlmTrainDataset._truncated_doc_ids`` (``--doc_budget_cr``).
3. HOW THE BUDGET IS HIT. The knob is searched; the OUTPUT IS NEVER CUT.
   Truncating the compressed string to the budget (the obvious first idea)
   is WRONG: LLMLingua-2 emits its retained words in document order, so
   cutting the tail turns the published method into a front-biased selector —
   a handicap invented by us, in our own favour. Instead we search for the
   most generous selection that still fits.

   LLMLingua-2's own selection rule, inside ``PromptCompressor.__compress``,
   is: per chunk, ``threshold = percentile(token_weighted_word_probs, q)`` and
   keep every word with ``prob > threshold``. Its public ``rate`` knob reaches
   that rule only through ``q = int(100*(1-rate) + 1)`` — an INTEGER
   percentile. On these documents one percentile step moves ~5 words ~= 7
   decoder tokens, so a rate search can only land 0..7 tokens under a ~70
   (cr=8) or ~35 (cr=16) token budget: a 10-20% systematic underspend that
   would understate the published baseline.

   ``--search_mode threshold`` (default) therefore drives LLMLingua-2's OWN
   rule with a CONTINUOUS ``q``, bisected until the rendered text is the
   largest one fitting ``V_i``. Same scoring model, same per-chunk percentile
   rule, same renderer (``convert_tokens_to_string``) — only the quantisation
   of the knob is removed, which is exactly what the library's own
   ``target_token`` argument is trying (and failing) to express. Selection
   granularity becomes one word, so the residual shortfall is 0-2 tokens.
   Fidelity is not assumed: ``--fidelity_check_docs`` documents are compressed
   BOTH ways and the reimplemented renderer must reproduce the public API's
   ``compressed_prompt`` byte for byte, or the run aborts.

   Decoder BPE length can very rarely be non-monotone even though the retained
   word sets are nested. When an observed bisection probe detects that case,
   the implementation enumerates every percentile boundary at which a word
   set can change and chooses the lowest feasible percentile. This exact
   fallback reuses the same word scores and performs no extra model forward.

   ``--search_mode rate_grid`` keeps the pure public-API path (bisection on
   ``rate``) for settings the threshold path does not replicate — non-empty
   ``--force_tokens`` or ``--drop_consecutive``.

   Exact equality ``selected == V_i`` is still not guaranteed by either mode:
   the unit of selection is a whole word, worth 1..k decoder tokens, so the
   reachable counts are a ladder and the budget can fall between two rungs.
   What IS enforced, fatally, is ``selected <= V_i`` — the baseline can never
   get more slots than CVLM. The residual shortfall (fraction of samples below
   budget, mean deficit) is measured and reported so the write-up can state
   the real number instead of claiming a strict equality that does not hold.
   ``--strict_exact`` turns any shortfall into a fatal error.

``--force_tokens`` defaults to EMPTY on purpose. The LLMLingua-2 demo uses
``force_tokens=['\\n', '?']``, which is sane at rate 0.33; at the rates this
protocol demands (~0.12 at cr=8, ~0.06 at cr=16) forcing every newline would
spend a third to three quarters of the whole budget on newline tokens — the
PwC documents used here average ~25 newlines against a ~70 token budget at
cr=8. Keeping punctuation would be a self-inflicted handicap on the baseline,
not a faithful reproduction.

Documents repeat across rows (PwC asks several questions about one document),
and the compression is question-independent, so identical documents are
scored once and cached (``--no_cache`` disables). The stats report both wall
time and the per-document cost, so the un-cached cost is recoverable.

Output JSONL (one line per sample of the filtered dataset, idx = position in
that filtered dataset — the contract ``CvlmTrainDataset._load_doc_selection``
validates). The loader only checks the line count and the idx range, which
would not catch a permuted or stale file, so every line also carries the
provenance ``tools/audit_selection_jsonl.py`` re-checks against the dataset::

    {"idx": int, "selected_text": str,
     "source_sha256": str, "prompt_sha256": str,
     "selected_smol_tokens": int, "target_budget": int, "cr": int,
     "llmlingua_model": str, "model_revision": str,
     "l_enc": int, "l_dec": int, "percentile_q": float, "n_calls": int,
     "truncated": bool, "compress_seconds": float}

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
from typing import Callable, Dict, List, Sequence, Tuple

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
        help="Decoder-token budget V = max(min(ceil(L_enc/N), max_vision_len), 1) — the CVLM "
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
        "default — see the module docstring: at these budgets forcing punctuation is a handicap. "
        "Non-empty requires --search_mode rate_grid.",
    )
    p.add_argument(
        "--drop_consecutive", action="store_true", help="Drop repeated force_tokens (rate_grid mode only)."
    )
    p.add_argument(
        "--search_mode",
        type=str,
        default="threshold",
        choices=["threshold", "rate_grid"],
        help="threshold = bisect LLMLingua-2's own per-chunk percentile continuously (word-level "
        "granularity, one forward pass per document, verified byte-identical to the public API); "
        "rate_grid = bisect the public `rate` knob (1%% percentile quantisation).",
    )
    p.add_argument(
        "--search_iters",
        type=int,
        default=32,
        help="Bisection steps. threshold mode: pure CPU re-rendering. rate_grid mode: one "
        "LLMLingua forward per step, so keep it small (12 is plenty at 1%% granularity).",
    )
    p.add_argument(
        "--fidelity_check_docs",
        type=int,
        default=25,
        help="threshold mode: documents on which the reimplemented renderer must reproduce the "
        "public compress_prompt_llmlingua2 output byte for byte. 0 disables (not recommended).",
    )
    p.add_argument(
        "--strict_exact",
        action="store_true",
        help="Treat any sample landing BELOW the budget as fatal. Off by default: word-level "
        "selection granularity makes exact equality unreachable for some documents.",
    )
    p.add_argument("--no_cache", action="store_true", help="Do not reuse compressions of identical documents.")
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--stats_json", type=str, default="", help="Optional JSON with budget/timing statistics.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_revision(model_name: str) -> str:
    """Commit sha of the locally cached snapshot ('' when undeterminable)."""
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(model_name, local_files_only=True)
        return os.path.basename(os.path.normpath(path))
    except Exception:  # offline / non-hub path / hub API change
        return ""


def truncate_to_budget(dec_tok, text: str, budget: int) -> Tuple[str, int]:
    """SAFETY NET ONLY — cut `text` to at most `budget` decoder tokens.

    Never used on the normal path (both search modes return an already
    feasible string); it exists so a pathological document cannot produce an
    over-budget line, which the eval would consume without noticing.
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


class LinguaWordScorer:
    """LLMLingua-2's per-chunk word scoring, with the knob left continuous.

    Mirrors ``PromptCompressor._PromptCompressor__compress`` for the
    ``force_tokens=[] / drop_consecutive=False`` case: same chunking, same
    ``TokenClfDataset`` construction (including the library's quirk of leaving
    ``model_name`` at its BERT default), same token->word merge, same
    token-weighted percentile threshold, same ``convert_tokens_to_string``
    rendering and same ``"".join`` of chunks. The single difference is that
    ``render`` takes the percentile ``q`` as a float instead of receiving
    ``int(100*reduce_rate + 1)``.

    ``score`` runs the transformer ONCE per document; ``render`` is pure CPU,
    which is what makes an exact budget search affordable.
    """

    def __init__(self, compressor) -> None:
        self.c = compressor
        missing = [
            n
            for n in ("_PromptCompressor__chunk_context", "_PromptCompressor__merge_token_to_word",
                      "_PromptCompressor__token_prob_to_word_prob")
            if not hasattr(compressor, n)
        ]
        if missing:
            raise RuntimeError(
                f"llmlingua internals not found ({missing}); this tool is written against "
                f"llmlingua==0.2.2. Re-run with --search_mode rate_grid or pin the version."
            )
        self._chunk = getattr(compressor, "_PromptCompressor__chunk_context")
        self._merge = getattr(compressor, "_PromptCompressor__merge_token_to_word")
        self._t2w = getattr(compressor, "_PromptCompressor__token_prob_to_word_prob")

    def score(self, text: str, chunk_end_tokens: Sequence[str] = (".", "\n")) -> List[Tuple[List[str], List[float], List[float]]]:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader

        from llmlingua.utils import TokenClfDataset

        chunks = self._chunk(text, chunk_end_tokens=set(chunk_end_tokens))
        ds = TokenClfDataset(chunks, tokenizer=self.c.tokenizer, max_len=self.c.max_seq_len)
        dl = DataLoader(ds, batch_size=self.c.max_batch_size, shuffle=False, drop_last=False)
        out: List[Tuple[List[str], List[float], List[float]]] = []
        with torch.no_grad():
            for batch in dl:
                ids = batch["ids"].to(self.c.device, dtype=torch.long)
                mask = batch["mask"].to(self.c.device, dtype=torch.long) == 1
                logits = self.c.model(input_ids=ids, attention_mask=mask).logits
                probs = F.softmax(logits, dim=-1)
                for j in range(ids.shape[0]):
                    active_probs = torch.masked_select(probs[j, :, 1], mask[j])
                    active_ids = torch.masked_select(ids[j], mask[j])
                    tokens = self.c.tokenizer.convert_ids_to_tokens(active_ids.squeeze().tolist())
                    token_probs = [p for p in active_probs.cpu().numpy()]
                    words, valid_token_probs, _ = self._merge(
                        tokens=tokens,
                        token_probs=token_probs,
                        force_tokens=[],
                        token_map={},
                        force_reserve_digit=False,
                    )
                    word_probs = self._t2w(valid_token_probs, convert_mode="mean")
                    new_token_probs: List[float] = []
                    for word, word_prob in zip(words, word_probs):
                        n_tok = len(self.c.oai_tokenizer.encode(word))
                        new_token_probs.extend([word_prob for _ in range(n_tok)])
                    out.append((list(words), list(word_probs), new_token_probs))
        return out

    def render(self, scored: List[Tuple[List[str], List[float], List[float]]], q: float) -> str:
        import numpy as np

        q = min(max(float(q), 0.0), 100.0)
        parts: List[str] = []
        for words, word_probs, new_token_probs in scored:
            if not words:
                parts.append("")
                continue
            threshold = np.percentile(new_token_probs, q)
            keep = [
                w
                for w, p in zip(words, word_probs)
                if p > threshold or (threshold == 1.0 and p == threshold)
            ]
            parts.append(self.c.tokenizer.convert_tokens_to_string(keep))
        return "".join(parts)


def threshold_q_candidates(
    scored: List[Tuple[List[str], List[float], List[float]]]
) -> List[float]:
    """All percentile regimes at which LLMLingua's retained words can change.

    ``numpy.percentile`` linearly interpolates a sorted vector at rank
    ``q/100 * (n-1)``. A keep-set transition can therefore occur only where
    that interpolation reaches one of the token-weighted word probabilities.
    Evaluating each such boundary and one point in every open interval covers
    every distinct retained-word set, including the equality behaviour at the
    boundary itself.
    """
    boundaries = {0.0, 100.0}
    for _, _, weighted_word_probs in scored:
        values = sorted(float(x) for x in weighted_word_probs)
        n = len(values)
        if n <= 1:
            continue
        for i, value in enumerate(values):
            if i == 0 or value != values[i - 1]:
                boundaries.add(100.0 * i / (n - 1))
    ordered = sorted(boundaries)
    candidates = set(ordered)
    for left, right in zip(ordered, ordered[1:]):
        candidates.add((left + right) / 2.0)
    return sorted(candidates)


def exact_threshold_selection(
    scored: List[Tuple[List[str], List[float], List[float]]],
    render: Callable[[List[Tuple[List[str], List[float], List[float]]], float], str],
    token_length: Callable[[str], int],
    budget: int,
) -> Tuple[str, int, float]:
    """Return the lowest-percentile (largest nested word set) feasible output."""
    for q in threshold_q_candidates(scored):
        text = render(scored, q)
        tokens = token_length(text)
        if tokens <= budget:
            return text, tokens, q
    raise RuntimeError(
        "LLMLingua exact threshold enumeration found no output within "
        f"the decoder-token budget={budget}"
    )


def main() -> None:
    args = parse_args()
    if args.budget_cr <= 0:
        raise ValueError("--budget_cr must be > 0")
    if args.max_vision_len <= 0:
        raise ValueError("--max_vision_len must be > 0")
    if args.search_iters < 1:
        raise ValueError("--search_iters must be >= 1")

    import numpy as np
    import torch
    from tqdm import tqdm

    from cvlm_dataset import CvlmTrainDataset

    torch.manual_seed(args.seed)

    force_tokens: List[str] = [
        t.replace("\\n", "\n").replace("\\t", "\t") for t in args.force_tokens.split(",") if t != ""
    ]
    if args.search_mode == "threshold" and (force_tokens or args.drop_consecutive):
        raise ValueError(
            "--search_mode threshold reimplements LLMLingua-2's selection only for "
            "force_tokens=[] / drop_consecutive=False. Use --search_mode rate_grid."
        )

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
    revision = resolve_revision(args.llmlingua_model)
    print(f"  snapshot revision: {revision or '<unknown>'}")
    scorer = LinguaWordScorer(compressor) if args.search_mode == "threshold" else None

    def n_dec(s: str) -> int:
        return len(dec_tok(s, add_special_tokens=False, truncation=False)["input_ids"])

    def public_compress(document: str, rate: float) -> str:
        return str(
            compressor.compress_prompt_llmlingua2(
                document,
                rate=rate,
                force_tokens=force_tokens,
                drop_consecutive=args.drop_consecutive,
            )["compressed_prompt"]
        )

    gpu_seconds = [0.0]  # accumulated LLMLingua forward time
    call_count = [0]
    nonmonotone_fallback_count = [0]

    def compress_threshold(document: str, budget: int) -> Dict[str, object]:
        """Bisect the continuous percentile; one transformer pass per document."""
        assert scorer is not None
        t0 = time.time()
        scored = scorer.score(document)
        gpu_seconds[0] += time.time() - t0
        call_count[0] += 1

        lo, hi = 0.0, 100.0  # lo: keeps most words; hi: sparsest output
        probes: List[Tuple[float, int]] = []
        text0 = scorer.render(scored, lo)
        tok0 = n_dec(text0)
        probes.append((lo, tok0))
        if tok0 <= budget:
            return {
                "selected_text": text0,
                "dec_tokens": tok0,
                "q": lo,
                "n_calls": 1,
                "truncated": False,
            }
        # q=100 is not assumed to be the empty string: upstream deliberately
        # retains p==1 words when the threshold is exactly one. It must be a
        # real feasible output, otherwise no continuous setting of the
        # published selection rule satisfies this budget and inventing an
        # empty fallback would no longer be LLMLingua-2.
        text100 = scorer.render(scored, hi)
        tok100 = n_dec(text100)
        probes.append((hi, tok100))
        if tok100 > budget:
            raise RuntimeError(
                "LLMLingua-2 has no threshold output within the decoder-token "
                f"budget: sparsest output has {tok100} tokens, budget={budget}"
            )
        best_text, best_tok, best_q = text100, tok100, hi
        for _ in range(args.search_iters):
            mid = (lo + hi) / 2.0
            text = scorer.render(scored, mid)
            tok = n_dec(text)
            probes.append((mid, tok))
            if tok <= budget:
                hi = mid
                if tok > best_tok or (tok == best_tok and mid < best_q) or not best_text:
                    best_text, best_tok, best_q = text, tok, mid
                if tok == budget:
                    break
            else:
                lo = mid
        # Bisection relies on decoder-token length being non-increasing as the
        # threshold rises. Word sets are nested by construction, but BPE
        # boundary effects could in principle violate token-length monotonicity.
        # Refuse to publish a falsely "maximal" selection if that happens on
        # any point the actual search observed.
        ordered = sorted(probes)
        nonmonotone = False
        for (q0, t0), (q1, t1) in zip(ordered, ordered[1:]):
            if t1 > t0:
                nonmonotone = True
                break
        if nonmonotone:
            # The retained word sets are still nested as q rises. Enumerating
            # every keep-set regime lets us select the lowest q whose actual
            # decoder-token length is within budget, without truncating the
            # published method's output or running its transformer again.
            nonmonotone_fallback_count[0] += 1
            text, tok, q = exact_threshold_selection(
                scored, scorer.render, n_dec, budget
            )
            return {
                "selected_text": text,
                "dec_tokens": tok,
                "q": q,
                "n_calls": 1,
                "truncated": False,
                "nonmonotone_fallback": True,
            }
        return {
            "selected_text": best_text,
            "dec_tokens": best_tok,
            "q": best_q,
            "n_calls": 1,
            "truncated": False,
            "nonmonotone_fallback": False,
        }

    def compress_rate_grid(document: str, budget: int) -> Dict[str, object]:
        """Public-API path: bisect `rate`, one LLMLingua forward per probe."""
        l_dec = n_dec(document)
        lo, hi = 0.0, 1.0  # lo: feasible (empty), hi: infeasible (whole doc)
        rate = min(1.0, budget / max(l_dec, 1))
        best_text, best_tok, best_rate = "", 0, 0.0
        have_feasible = False
        smallest_over_text, smallest_over_tok = "", 0
        n_calls = 0
        for _ in range(args.search_iters):
            t0 = time.time()
            text = public_compress(document, rate)
            gpu_seconds[0] += time.time() - t0
            call_count[0] += 1
            n_calls += 1
            tok = n_dec(text)
            if tok <= budget:
                if tok > best_tok or not have_feasible:
                    best_text, best_tok, best_rate = text, tok, rate
                have_feasible = True
                lo = max(lo, rate)
                if tok == budget:
                    break
                nxt = (rate + hi) / 2.0
            else:
                if not smallest_over_text or tok < smallest_over_tok:
                    smallest_over_text, smallest_over_tok = text, tok
                hi = min(hi, rate)
                nxt = (lo + rate) / 2.0
            if abs(nxt - rate) < 1e-5:
                break
            rate = nxt
        truncated = False
        if not have_feasible and smallest_over_text:
            best_text, best_tok = truncate_to_budget(dec_tok, smallest_over_text, budget)
            best_rate = hi
            truncated = True
        return {
            "selected_text": best_text,
            "dec_tokens": best_tok,
            "q": 100.0 * (1.0 - best_rate),
            "n_calls": n_calls,
            "truncated": truncated,
            "nonmonotone_fallback": False,
        }

    def compress_one(document: str, budget: int) -> Dict[str, object]:
        l_dec = n_dec(document)
        if l_dec <= budget:
            # Already inside the budget: compressing would only destroy
            # information the budget can afford. Same convention as the
            # truncation strategies (a short doc is kept whole).
            return {
                "selected_text": document,
                "dec_tokens": l_dec,
                "l_dec": l_dec,
                "q": 0.0,
                "n_calls": 0,
                "truncated": False,
                "nonmonotone_fallback": False,
            }
        t0 = time.time()
        out = compress_threshold(document, budget) if scorer else compress_rate_grid(document, budget)
        out["l_dec"] = l_dec
        out["compress_seconds"] = time.time() - t0
        return out

    # --- fidelity: the reimplemented renderer must match the public API ------
    n_fidelity = 0
    t_fidelity = time.time()
    if scorer is not None and args.fidelity_check_docs > 0:
        print(f"Fidelity check on up to {args.fidelity_check_docs} documents "
              f"(reimplemented renderer vs public compress_prompt_llmlingua2)...")
        seen: set = set()
        for idx in range(n_total):
            if n_fidelity >= args.fidelity_check_docs:
                break
            doc = str(dataset._hf[dataset._row_indices[idx]]["input"])
            if doc in seen:
                continue
            seen.add(doc)
            scored = scorer.score(doc)
            for rate in (0.3, 0.1):
                q = int(100 * (1.0 - rate) + 1)
                mine = scorer.render(scored, float(q))
                theirs = public_compress(doc, rate)
                if mine != theirs:
                    raise RuntimeError(
                        "FATAL: the reimplemented LLMLingua-2 renderer does not reproduce the "
                        f"public API at rate={rate} (doc sha {sha256_text(doc)[:12]}). "
                        f"len(mine)={len(mine)} len(theirs)={len(theirs)}. "
                        "Re-run with --search_mode rate_grid."
                    )
            n_fidelity += 1
        print(f"  fidelity OK on {n_fidelity} documents x 2 rates (byte-identical)")
    fidelity_seconds = time.time() - t_fidelity
    gpu_seconds[0] = 0.0  # the measured cost below is the SELECTION cost only
    call_count[0] = 0

    cache: Dict[str, Dict[str, object]] = {}
    budgets: List[int] = []
    achieved_all: List[int] = []
    n_cache_hits = 0
    n_empty = 0
    n_truncated = 0

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    # Never expose a half-written selection as a reusable result. A preempted
    # job may leave this temporary file behind, but the canonical JSONL only
    # appears after every row and every budget check has succeeded.
    tmp_output_jsonl = f"{args.output_jsonl}.tmp.{os.getpid()}"
    t_start = time.time()
    with open(tmp_output_jsonl, "w") as fh, torch.no_grad():
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
            if out["truncated"]:
                n_truncated += 1
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
            if args.strict_exact and achieved != budget:
                raise RuntimeError(
                    f"idx={idx}: --strict_exact set but the selection lands at {achieved} "
                    f"of V_i={budget} decoder tokens (word-level granularity)"
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
                "search_mode": args.search_mode,
                "l_enc": int(l_enc),
                "l_dec": int(out["l_dec"]),
                "percentile_q": float(out["q"]),
                "n_calls": int(out["n_calls"]),
                "truncated": bool(out["truncated"]),
                "nonmonotone_fallback": bool(out.get("nonmonotone_fallback", False)),
                "compress_seconds": float(out.get("compress_seconds", 0.0)),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            budgets.append(int(budget))
            achieved_all.append(int(achieved))

    wall = time.time() - t_start
    b = np.asarray(budgets, dtype=np.float64)
    a = np.asarray(achieved_all, dtype=np.float64)
    deficit = b - a
    under = deficit > 0
    n_docs = len(cache) if not args.no_cache else n_total
    stats = {
        "output_jsonl": args.output_jsonl,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "budget_cr": int(args.budget_cr),
        "max_vision_len": int(args.max_vision_len),
        "llmlingua_model": args.llmlingua_model,
        "model_revision": revision,
        "search_mode": args.search_mode,
        "search_iters": int(args.search_iters),
        "force_tokens": force_tokens,
        "fidelity_docs_checked": int(n_fidelity),
        "fidelity_seconds": float(fidelity_seconds),
        "n_samples": int(n_total),
        "n_unique_documents": int(n_docs),
        "n_cache_hits": int(n_cache_hits),
        "budget_mean": float(b.mean()),
        "budget_median": float(np.median(b)),
        "budget_min": int(b.min()),
        "budget_max": int(b.max()),
        "achieved_mean": float(a.mean()),
        "achieved_median": float(np.median(a)),
        "achieved_min": int(a.min()),
        "achieved_max": int(a.max()),
        "budget_utilisation_mean": float((a / np.maximum(b, 1)).mean()),
        "frac_exact_budget": float((deficit == 0).mean()),
        "frac_under_budget": float(under.mean()),
        "mean_deficit_all": float(deficit.mean()),
        "median_deficit_all": float(np.median(deficit)),
        "p90_deficit_all": float(np.percentile(deficit, 90)),
        "p95_deficit_all": float(np.percentile(deficit, 95)),
        "mean_deficit_when_under": float(deficit[under].mean()) if under.any() else 0.0,
        "max_deficit": int(deficit.max()),
        "frac_over_budget": float((a > b).mean()),  # must be 0.0
        "n_truncated_safety_net": int(n_truncated),
        "n_empty_selections": int(n_empty),
        "n_llmlingua_forward_calls": int(call_count[0]),
        "n_nonmonotone_fallback_documents": int(nonmonotone_fallback_count[0]),
        "llmlingua_seconds_total": float(gpu_seconds[0]),
        "llmlingua_seconds_per_document": float(gpu_seconds[0] / max(n_docs, 1)),
        "wall_seconds_total": float(wall),
        "wall_seconds_per_sample": float(wall / max(n_total, 1)),
        # What the same rows would have cost with the document cache OFF
        # (PwC repeats each document across ~15 questions).
        "estimated_seconds_without_cache": float(wall / max(n_docs, 1) * n_total),
    }
    if stats["frac_over_budget"] != 0.0:
        raise RuntimeError("BUG: some selections exceed the decoder-token budget")
    tmp_stats_json = ""
    if args.stats_json:
        os.makedirs(os.path.dirname(args.stats_json) or ".", exist_ok=True)
        tmp_stats_json = f"{args.stats_json}.tmp.{os.getpid()}"
        with open(tmp_stats_json, "w") as fh:
            json.dump(stats, fh, indent=2)

    # Publish only after both complete artefacts have been written and all
    # aggregate checks passed. The sbatch requires the pair on resume.
    os.replace(tmp_output_jsonl, args.output_jsonl)
    if tmp_stats_json:
        os.replace(tmp_stats_json, args.stats_json)

    print(f"\nWrote {n_total} lines to {args.output_jsonl}")
    for k, v in stats.items():
        print(f"  {k:34s}: {v}")
    if args.stats_json:
        print(f"Stats -> {args.stats_json}")
    print(
        "Consume with: python src/eval_cvlm.py --mode baseline_llm_full|sft "
        f"--doc_truncation file --doc_selection_file {args.output_jsonl} "
        "(same dataset/filter flags!)"
    )


if __name__ == "__main__":
    main()
