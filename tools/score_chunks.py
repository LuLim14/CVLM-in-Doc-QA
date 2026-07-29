#!/usr/bin/env python3
"""Skim-then-read, stage 1: offline question-conditioned chunk scoring.

A trained CVLM compressor "skims" the compressed document: the decoder's
attention over the vision tokens is a FREE question-conditioned importance
estimate for the underlying encoder chunks. Each vision token j is the pooled
image of `compression_rate` contiguous encoder tokens (chunk j), so an
importance vector over vision tokens is an importance vector over document
chunks. The top chunks (under a decoder-token budget) are then glued back
together as RAW text ("read" stage) and fed to the frozen decoder via
eval_cvlm.py --doc_truncation=file --doc_selection_file <output_jsonl>.

Causality note (why the question is fed twice)
----------------------------------------------
CVLM's decoder layout is [prompt, vision, answer]: the question PRECEDES the
vision span, so under the causal mask the attention "from question positions
to vision positions" is identically zero. The probe therefore runs one forward
with

    inputs_embeds = [prompt_embs, vision_embeds, prompt_embs]

i.e. the question is re-appended AFTER the vision block — exactly where answer
tokens sat during training — and importance is aggregated from the TRAILING
question copy's attention onto the vision block. RoPE is relative, so the
(query -> vision) distances of the trailing copy match the trained
(answer -> vision) distances, and the [prompt, vision] prefix stays exactly as
trained.

Scoring
-------
attentions[l][0, h, q, k] are post-softmax weights. We average over heads and
over the selected layers (--layers last4|all), then SUM over the trailing
question-copy query positions and keep the vision-key columns:

    importance[j] = sum_{q in question copy} mean_{l,h} A[l,h,q, P+j],  j<V

Chunk -> char span: the FAST encoder tokenizer's offset_mapping maps chunk j
(encoder tokens [j*cr, (j+1)*cr)) to a character span of the document. The
decoder tokenizer is slow (use_fast=False, no offsets), so ALL span logic runs
on encoder offsets + char substrings; decoder tokens are only counted by
tokenizing substrings.

Budget: B = max(min(ceil(L_enc / budget_cr), max_vision_len), 1) decoder
tokens — the same formula eval_cvlm.py --doc_truncation first/last/random/
oracle uses (--doc_budget_cr), so 'file' rows are budget-comparable. Greedy
selection by descending importance; a chunk that does not fit the remaining
budget is skipped (cheaper, less important chunks may still fit); if nothing
fits, the single top chunk is taken (may overshoot B). Selected chunks are
rendered in DOCUMENT order; maximal runs of adjacent chunks become one
substring, runs are joined with " ... ".

Output JSONL (one line per sample of the filtered dataset, same order/idx as
CvlmTrainDataset with identical filters):

    {"idx": int, "selected_text": str, "n_chunks": int,
     "coverage_tokens": int, "scores_entropy": float}

coverage_tokens = sum of the per-chunk decoder-token costs the greedy loop
accounted for (merging + " ... " separators can shift the final tokenization
by a few tokens). scores_entropy = entropy (nats) of the normalised
importance distribution over the V chunks — low entropy = peaked saliency.

Example:
    python tools/score_chunks.py \
        --checkpoint_path ckpts/run/model_step_20000.safetensors \
        --model_name_or_path HuggingFaceTB/SmolLM2-1.7B \
        --text_encoder_name answerdotai/ModernBERT-base \
        --compression_rate 8 --num_pool_latents 16 --max_vision_len 512 \
        --no_vit true \
        --dataset_split test --max_samples 2000 \
        --layers last4 --budget_cr 8 \
        --output_jsonl results/skim_cr8.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List, Sequence, Tuple

# The tool only reads attentions — never computes a loss — so liger's fused
# forward patches buy nothing and are one more moving part; default them off
# (respects an explicit DISABLE_LIGER=0).
os.environ.setdefault("DISABLE_LIGER", "1")

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ---------------------------------------------------------------------------
# Pure helpers (no torch/transformers — unit-testable with stub inputs)
# ---------------------------------------------------------------------------


def chunk_char_spans(
    offsets: Sequence[Sequence[int]], compression_rate: int, max_chunks: int
) -> List[Tuple[int, int]]:
    """Map contiguous `compression_rate`-token chunks to document char spans.

    Mirrors _chunked_attention_pool geometry: chunk j covers encoder tokens
    [j*cr, min((j+1)*cr, L)); at most `max_chunks` chunks (the pool truncates
    docs needing more than max_vision_len chunks, dropping the tail tokens).

    Args:
        offsets: fast-tokenizer offset_mapping [(char_start, char_end), ...]
            for the document, add_special_tokens=False (zero-width special
            offsets would corrupt spans).
        compression_rate: encoder tokens per chunk (cr).
        max_chunks: hard cap on chunk count (max_vision_len).

    Returns:
        [(char_start, char_end), ...] — one span per chunk, chunk order.
    """
    cr = max(int(compression_rate), 1)
    n = len(offsets)
    if n == 0:
        return []
    n_chunks = min((n + cr - 1) // cr, max(int(max_chunks), 1))
    spans: List[Tuple[int, int]] = []
    for j in range(n_chunks):
        t0 = j * cr
        t1 = min(t0 + cr, n) - 1
        spans.append((int(offsets[t0][0]), int(offsets[t1][1])))
    return spans


def select_chunks_by_budget(
    importance: Sequence[float], chunk_costs: Sequence[int], budget: int
) -> Tuple[List[int], int]:
    """Greedy chunk selection under a decoder-token budget.

    Visits chunks by descending importance (ties -> document order); a chunk is
    taken iff it fits the remaining budget, otherwise skipped (a later, cheaper
    chunk may still fit). If no chunk fits at all, the single top-importance
    chunk is taken anyway so the selection is never empty (may overshoot).

    Returns:
        (chosen_indices_sorted_in_document_order, coverage_tokens)
    """
    n = len(importance)
    assert len(chunk_costs) == n, f"importance ({n}) vs chunk_costs ({len(chunk_costs)}) length mismatch"
    order = sorted(range(n), key=lambda j: (-float(importance[j]), j))
    chosen: List[int] = []
    used = 0
    for j in order:
        c = int(chunk_costs[j])
        if used + c <= budget:
            chosen.append(j)
            used += c
    if not chosen and order:
        top = order[0]
        chosen = [top]
        used = int(chunk_costs[top])
    return sorted(chosen), used


def render_selection(
    document: str, spans: Sequence[Tuple[int, int]], chosen_sorted: Sequence[int]
) -> str:
    """Glue chosen chunks back into raw text, in DOCUMENT order.

    Maximal runs of adjacent chunk indices become a single char-span substring
    (so intra-run whitespace/punctuation between tokens is preserved verbatim);
    non-adjacent runs are joined with " ... ".
    """
    if not chosen_sorted:
        return ""
    runs: List[List[int]] = []
    for j in chosen_sorted:
        if runs and j == runs[-1][1] + 1:
            runs[-1][1] = j
        else:
            runs.append([j, j])
    return " ... ".join(document[spans[a][0] : spans[b][1]] for a, b in runs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Skim-then-read: score document chunks via CVLM decoder attention over vision tokens"
    )
    p.add_argument("--checkpoint_path", type=str, required=True, help="Path to model_step_*.safetensors")
    # Model flags — keep 1:1 with eval_cvlm.py so the same checkpoint loads the
    # same way here.
    p.add_argument("--model_name_or_path", type=str, default=None)
    p.add_argument("--vision_encoder_name", type=str, default=None)
    p.add_argument("--text_encoder_name", type=str, default=None)
    p.add_argument("--compression_rate", type=int, default=4)
    p.add_argument(
        "--num_pool_latents",
        type=int,
        default=int(os.environ.get("NUM_POOL_LATENTS", "1")),
        help="K learnable latent queries of the chunked attention pool; must match training.",
    )
    p.add_argument("--max_vision_len", type=int, default=512)
    p.add_argument("--max_source_len", type=int, default=0, help="0 = compression_rate * max_vision_len")
    p.add_argument(
        "--legacy_projector",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("LEGACY_PROJECTOR", "0") == "1",
        help="Must match training: omit pre-MLP RMSNorm so old checkpoints load without extra keys.",
    )
    p.add_argument(
        "--vit_init",
        type=str,
        choices=["pretrained", "random"],
        default=os.environ.get("VIT_INIT", "pretrained"),
        help="Must match training.",
    )
    p.add_argument(
        "--freeze_vit",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("FREEZE_VIT", "0") == "1",
        help="Must match training (frozen ViT weights reload from HF, they are absent from the checkpoint).",
    )
    p.add_argument(
        "--no_vit",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("NO_VIT", "0") == "1",
        help="MUST match training for the no-ViT branch (vision embeds bypass the ViT in _encode_vision).",
    )
    # Dataset flags — MUST match the eval_cvlm.py run that will consume the
    # JSONL: idx alignment is by position in the filtered dataset.
    p.add_argument("--dataset_name", type=str, default="sggetao/PwC")
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--max_prompt_len", type=int, default=512)
    p.add_argument("--max_answer_len", type=int, default=2048)
    p.add_argument("--max_samples", type=int, default=0, help="Limit to first N samples (0 = all).")
    # Scoring / selection flags.
    p.add_argument(
        "--layers",
        type=str,
        default="last4",
        choices=["last4", "all"],
        help="Which decoder layers' attentions to average: last 4 or all.",
    )
    p.add_argument(
        "--budget_cr",
        type=int,
        required=True,
        help="Decoder-token budget B = max(min(ceil(L_enc/N), max_vision_len), 1) — the CVLM "
        "vision-token budget at compression_rate=N; same formula as eval_cvlm.py --doc_budget_cr.",
    )
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.budget_cr <= 0:
        raise ValueError("--budget_cr must be > 0")

    # Heavy imports deferred so the pure helpers above import torch-free (tests).
    import torch
    from safetensors.torch import load_file
    from tqdm import tqdm

    from cvlm_dataset import CvlmTrainDataset
    from modeling import CVLM, ModelArguments, TrainingArguments

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and not args.no_bf16

    model_args = ModelArguments(train=False)
    if args.model_name_or_path:
        model_args.model_name_or_path = args.model_name_or_path
    if args.vision_encoder_name:
        model_args.vision_encoder_name = args.vision_encoder_name
    if args.text_encoder_name:
        model_args.text_encoder_name = args.text_encoder_name
    model_args.max_vision_len = args.max_vision_len
    model_args.compression_rate = args.compression_rate
    model_args.num_pool_latents = max(int(args.num_pool_latents), 1)
    model_args.projector_use_input_rmsnorm = not bool(args.legacy_projector)
    model_args.vit_init = str(args.vit_init)
    model_args.vit_freeze = bool(args.freeze_vit)
    model_args.vit_skip = bool(args.no_vit)

    training_args = TrainingArguments(output_dir="/tmp/score_chunks_dummy")
    training_args.bf16 = bool(use_bf16)

    print(f"Loading model ({model_args.model_name_or_path})...")
    model = CVLM(model_args, training_args)
    model.to(device)
    print(f"Loading checkpoint: {args.checkpoint_path}")
    state_dict = load_file(args.checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  loaded; missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    # output_attentions needs real attention probabilities; sdpa/flash return
    # None, so force the eager path on the decoder.
    try:
        model.decoder.set_attn_implementation("eager")
    except (AttributeError, TypeError, ValueError):
        model.decoder.config._attn_implementation = "eager"

    # Base transformer (no lm_head): attentions are identical and the [S, vocab]
    # logits matmul is skipped entirely.
    try:
        base_decoder = model.decoder.get_decoder()
    except AttributeError:
        base_decoder = getattr(model.decoder, "model", model.decoder)

    max_source_len = args.max_source_len if args.max_source_len > 0 else args.compression_rate * args.max_vision_len
    dataset = CvlmTrainDataset(
        hf_dataset_name=args.dataset_name,
        hf_split=args.dataset_split,
        decoder_tokenizer_name=model_args.model_name_or_path,
        encoder_tokenizer_name=model_args.text_encoder_name,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
        max_source_len=max_source_len,
        max_samples=args.max_samples,
    )

    enc_tok = dataset._enc_tok  # FAST — has offset_mapping
    dec_tok = dataset._dec_tok  # slow — only used to COUNT tokens of substrings
    cr = max(int(args.compression_rate), 1)
    use_last4 = args.layers == "last4"

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    n_total = len(dataset)
    sum_chunks = 0
    sum_cov = 0
    sum_ent = 0.0

    with open(args.output_jsonl, "w") as fh, torch.no_grad():
        for idx in tqdm(range(n_total), desc=f"score chunks ({args.layers}, budget_cr={args.budget_cr})"):
            record = dataset._hf[dataset._row_indices[idx]]
            document = str(record["input"])
            sample = dataset[idx]
            source_ids = sample["source_ids"].unsqueeze(0).to(device)
            source_mask = torch.ones_like(source_ids)  # single sample: no padding
            prompt_ids = sample["prompt_ids"].unsqueeze(0).to(device)

            # vit_skip / vit_freeze / pretrained are all handled inside
            # _encode_vision (vision_encoder=None bypasses the ViT block).
            vision_embeds, vision_mask = model._encode_vision(source_ids, source_mask)
            V = vision_embeds.size(1)
            P = prompt_ids.size(1)
            assert int(vision_mask.sum().item()) == V, "unpadded single sample must have all-real vision slots"

            embed_layer = model.decoder.get_input_embeddings()
            prompt_embs = embed_layer(prompt_ids)
            # [prompt, vision, prompt-copy]: trained prefix + question re-asked
            # after the vision block (see module docstring, causality note).
            dec_in = torch.cat([prompt_embs, vision_embeds, prompt_embs], dim=1)
            attn_mask = torch.ones((1, dec_in.size(1)), dtype=torch.long, device=device)
            out = base_decoder(
                inputs_embeds=dec_in,
                attention_mask=attn_mask,
                output_attentions=True,
                use_cache=False,
            )
            attns = out.attentions  # n_layers x [1, n_heads, S, S]
            sel = attns[-4:] if use_last4 else attns
            # Mean over layers and heads -> [S, S] (rows = queries, cols = keys).
            A = torch.stack([a[0].float() for a in sel], dim=0).mean(dim=(0, 1))
            # Sum over the trailing question-copy queries, keep vision keys.
            importance = A[P + V :, P : P + V].sum(dim=0)  # [V]

            tot = float(importance.sum().item())
            if tot > 0:
                probs = (importance / tot).clamp_min(1e-12)
                entropy = float(-(probs * probs.log()).sum().item())
            else:
                # Degenerate (e.g. empty prompt): uniform tie -> uniform entropy.
                entropy = math.log(max(V, 1))

            # Chunk j -> char span via the FAST encoder tokenizer (same
            # truncation as dataset __getitem__, so token count matches V).
            enc = enc_tok(
                document,
                add_special_tokens=False,
                truncation=True,
                max_length=dataset.max_source_len,
                return_offsets_mapping=True,
            )
            spans = chunk_char_spans(enc["offset_mapping"], cr, model.max_vision_len)
            if len(spans) != V:
                raise RuntimeError(
                    f"idx={idx}: {len(spans)} char-span chunks vs V={V} vision tokens. "
                    f"Offset-mapping tokenization diverged from dataset source_ids "
                    f"(cr={cr}, max_vision_len={model.max_vision_len}, "
                    f"max_source_len={dataset.max_source_len})."
                )

            # Budget: same formula (and same L_enc measurement) as
            # CvlmTrainDataset._truncated_doc_ids with doc_budget_cr=budget_cr.
            l_enc = min(
                len(enc_tok(document, add_special_tokens=False, truncation=False)["input_ids"]),
                dataset.max_source_len,
            )
            budget = max(min((l_enc + args.budget_cr - 1) // args.budget_cr, args.max_vision_len), 1)

            chunk_costs = [
                len(dec_tok(document[s:e], add_special_tokens=False, truncation=False)["input_ids"])
                if e > s
                else 0
                for (s, e) in spans
            ]
            chosen, coverage = select_chunks_by_budget(importance.tolist(), chunk_costs, budget)
            selected_text = render_selection(document, spans, chosen)

            rec = {
                "idx": idx,
                "selected_text": selected_text,
                "n_chunks": len(chosen),
                "coverage_tokens": int(coverage),
                "scores_entropy": entropy,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            sum_chunks += len(chosen)
            sum_cov += int(coverage)
            sum_ent += entropy

    n = max(n_total, 1)
    print(
        f"\nWrote {n_total} lines to {args.output_jsonl}\n"
        f"  mean n_chunks        : {sum_chunks / n:.2f}\n"
        f"  mean coverage_tokens : {sum_cov / n:.1f}\n"
        f"  mean scores_entropy  : {sum_ent / n:.3f} nats"
    )
    print(
        "Consume with: python src/eval_cvlm.py --mode baseline_llm_full "
        f"--doc_truncation file --doc_selection_file {args.output_jsonl} "
        "(same dataset/filter flags!)"
    )


if __name__ == "__main__":
    main()
