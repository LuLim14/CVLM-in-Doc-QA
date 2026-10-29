"""Unit tests for the MIXED-FIDELITY training mode.

Covers the three pure pieces (no HF downloads, CPU only):
  1. cvlm_dataset.compute_chunk_decoder_spans — encoder-chunk -> decoder-token
     span mapping via char offsets (synthetic offset tables, tiny "vocab").
  2. modeling._sample_raw_chunk_flags — Bernoulli(p) raw selection + unbiased
     budget trim.
  3. modeling._assemble_mixed_decoder_input — block order (DOCUMENT ORDER),
     attention masks, and labels of the assembled decoder input.
Plus the collator passthrough of the mixed extras.
"""

import random

import torch

from cvlm_dataset import compute_chunk_decoder_spans, make_collate_fn
from modeling import _assemble_mixed_decoder_input, _sample_raw_chunk_flags

# ---------------------------------------------------------------------------
# compute_chunk_decoder_spans
# ---------------------------------------------------------------------------


def test_chunk_spans_basic_contiguous():
    # 12-char doc; encoder tokens are 2 chars each (6 tokens), decoder tokens
    # are 3 chars each (4 tokens). cr=2 -> 3 chunks with char starts 0, 4, 8.
    enc = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]
    dec = [(0, 3), (3, 6), (6, 9), (9, 12)]
    spans = compute_chunk_decoder_spans(enc, dec, compression_rate=2, max_chunks=8)
    assert spans == [(0, 1), (1, 2), (2, 4)]
    # Contiguous, non-decreasing, covers all 4 decoder tokens exactly once.
    for (s0, e0), (s1, _) in zip(spans, spans[1:]):
        assert e0 == s1
        assert e0 >= s0
    assert spans[0][0] == 0 and spans[-1][1] == len(dec)


def test_chunk_spans_max_chunks_cap_drops_tail_decoder_tokens():
    enc = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]
    dec = [(0, 3), (3, 6), (6, 9), (9, 12)]
    spans = compute_chunk_decoder_spans(enc, dec, compression_rate=2, max_chunks=2)
    assert len(spans) == 2
    # Chunk 2 covers enc chars [0,8); dec token 3 (chars 9..12) must be dropped.
    assert spans[-1][1] <= 3


def test_chunk_spans_empty_when_decoder_token_swallows_chunks():
    # One decoder token spans the whole 4-char doc; 4 encoder tokens, cr=1.
    # The single decoder token is assigned to the LAST chunk whose char start
    # it covers; earlier chunks get empty spans (callers keep them compressed).
    enc = [(0, 1), (1, 2), (2, 3), (3, 4)]
    dec = [(0, 4)]
    spans = compute_chunk_decoder_spans(enc, dec, compression_rate=1, max_chunks=8)
    assert spans == [(0, 0), (0, 0), (0, 0), (0, 1)]


def test_chunk_spans_empty_inputs():
    assert compute_chunk_decoder_spans([], [(0, 3)], 2, 8) == []
    assert compute_chunk_decoder_spans([(0, 2)], [], 2, 8) == []


# ---------------------------------------------------------------------------
# _sample_raw_chunk_flags
# ---------------------------------------------------------------------------


def test_sample_raw_flags_p_zero_all_compressed():
    flags = _sample_raw_chunk_flags([3, 4, 5], p=0.0, budget=100, rng=random.Random(0))
    assert flags == [False, False, False]


def test_sample_raw_flags_p_one_all_raw_when_budget_allows():
    flags = _sample_raw_chunk_flags([3, 4, 5], p=1.0, budget=100, rng=random.Random(0))
    assert flags == [True, True, True]


def test_sample_raw_flags_zero_length_chunk_never_raw():
    flags = _sample_raw_chunk_flags([3, 0, 5], p=1.0, budget=100, rng=random.Random(0))
    assert flags == [True, False, True]


def test_sample_raw_flags_budget_trims_cost():
    lens = [5, 5, 5]
    flags = _sample_raw_chunk_flags(lens, p=1.0, budget=7, rng=random.Random(0))
    cost = sum(lens[i] if flags[i] else 1 for i in range(len(lens)))
    assert cost <= 7
    # 15 -> 11 -> 7: exactly one raw chunk survives.
    assert sum(flags) == 1


def test_sample_raw_flags_budget_never_below_all_compressed():
    # budget == n always fits the all-compressed configuration.
    lens = [9, 9, 9, 9]
    flags = _sample_raw_chunk_flags(lens, p=1.0, budget=4, rng=random.Random(1))
    cost = sum(lens[i] if flags[i] else 1 for i in range(len(lens)))
    assert cost <= 4
    assert flags == [False, False, False, False]


# ---------------------------------------------------------------------------
# _assemble_mixed_decoder_input — stub embeddings with a tiny "vocab" where the
# embedding VALUE encodes the identity of the block, so ordering is checkable.
# ---------------------------------------------------------------------------


def _stub_inputs():
    B, P, V, A, T, D = 2, 2, 3, 2, 6, 3
    # doc token t of sample b -> constant vector 100*b + t.
    doc_embs = torch.zeros(B, T, D)
    for b in range(B):
        for t in range(T):
            doc_embs[b, t] = 100 * b + t
    # vision chunk j of sample b -> constant vector 1000 + 100*b + j.
    vision_embeds = torch.zeros(B, V, D)
    for b in range(B):
        for j in range(V):
            vision_embeds[b, j] = 1000 + 100 * b + j
    vision_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
    prompt_embs = torch.full((B, P, D), -1.0)
    prompt_mask = torch.tensor([[0, 1], [1, 1]], dtype=torch.long)  # left-padded
    answer_embs = torch.full((B, A, D), -2.0)
    answer_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.long)  # right-padded
    labels_src = torch.tensor([[7, 8], [9, -100]], dtype=torch.long)
    chunk_spans = torch.tensor(
        [
            [[0, 2], [2, 3], [3, 6]],  # sample 0: 3 chunks
            [[0, 0], [1, 2], [0, 0]],  # sample 1: 2 chunks (chunk 0 span EMPTY)
        ],
        dtype=torch.long,
    )
    chunk_counts = torch.tensor([3, 2], dtype=torch.long)
    return (
        prompt_embs,
        prompt_mask,
        vision_embeds,
        vision_mask,
        doc_embs,
        chunk_spans,
        chunk_counts,
        answer_embs,
        answer_mask,
        labels_src,
    )


def test_assemble_mixed_block_order_and_values():
    (
        prompt_embs,
        prompt_mask,
        vision_embeds,
        vision_mask,
        doc_embs,
        chunk_spans,
        chunk_counts,
        answer_embs,
        answer_mask,
        labels_src,
    ) = _stub_inputs()
    # Sample 0: raw, compressed, raw. Sample 1: raw-but-empty-span (falls back
    # to compressed), raw.
    raw_flags = [[True, False, True], [True, True]]
    dec_in, attn, labels, stats = _assemble_mixed_decoder_input(
        prompt_embs,
        prompt_mask,
        vision_embeds,
        vision_mask,
        doc_embs,
        chunk_spans,
        chunk_counts,
        raw_flags,
        answer_embs,
        answer_mask,
        labels_src,
    )
    P, A = 2, 2
    # Sample 0 mixed segment, DOCUMENT ORDER:
    #   doc[0:2] (raw) | vision[1] (compressed) | doc[3:6] (raw)  -> length 6.
    #   (vision chunk j of sample 0 encodes as 1000 + 0*100 + j.)
    expected0 = [0.0, 1.0, 1001.0, 3.0, 4.0, 5.0]
    got0 = [float(dec_in[0, P + k, 0]) for k in range(6)]
    assert got0 == expected0
    # Sample 1 mixed segment: vision[0] (empty-span fallback) | doc[1:2] (raw).
    expected1 = [1100.0, 101.0]
    got1 = [float(dec_in[1, P + k, 0]) for k in range(2)]
    assert got1 == expected1
    assert stats["mixed_lens"] == [6, 2]
    assert stats["raw_counts"] == [2, 1]
    assert stats["n_chunks"] == [3, 2]
    # Geometry: [P | M_max=6 | A].
    assert dec_in.shape == (2, P + 6 + A, 3)
    assert attn.shape == (2, P + 6 + A)
    assert labels.shape == (2, P + 6 + A)


def test_assemble_mixed_masks_and_padding():
    args = _stub_inputs()
    raw_flags = [[True, False, True], [False, False]]
    dec_in, attn, labels, stats = _assemble_mixed_decoder_input(*args[:7], raw_flags, *args[7:])
    prompt_mask, answer_mask = args[1], args[8]
    P, M, A = 2, 6, 2
    # Prompt/answer sub-masks are passed through untouched.
    assert torch.equal(attn[:, :P], prompt_mask)
    assert torch.equal(attn[:, P + M:], answer_mask)
    # Sample 0 mixed block fully real; sample 1 has 2 real + 4 pad slots.
    assert attn[0, P:P + M].tolist() == [1, 1, 1, 1, 1, 1]
    assert attn[1, P:P + M].tolist() == [1, 1, 0, 0, 0, 0]
    # Padded mixed positions are zero embeddings.
    assert torch.all(dec_in[1, P + 2:P + M] == 0)
    assert stats["mixed_lens"] == [6, 2]


def test_assemble_mixed_labels_answer_only():
    args = _stub_inputs()
    labels_src = args[9]
    raw_flags = [[False, True, False], [True, False]]
    _, _, labels, stats = _assemble_mixed_decoder_input(*args[:7], raw_flags, *args[7:])
    P = 2
    M = max(stats["mixed_lens"])  # mixed block is padded to the batch max
    # Sample 0: vision | doc[2:3] | vision -> 3; sample 1: vision (empty-span
    # fallback) | vision -> 2.
    assert stats["mixed_lens"] == [3, 2]
    # Loss must be masked over prompt + mixed block (both raw and compressed
    # chunks!) and carried only on the answer part.
    assert torch.all(labels[:, :P + M] == -100)
    assert torch.equal(labels[:, P + M:], labels_src)


def test_assemble_all_compressed_matches_vision_order():
    """p=0-like flags: mixed block degenerates to the legacy vision sequence."""
    args = _stub_inputs()
    raw_flags = [[False, False, False], [False, False]]
    dec_in, attn, _, stats = _assemble_mixed_decoder_input(*args[:7], raw_flags, *args[7:])
    P = 2
    assert [float(dec_in[0, P + k, 0]) for k in range(3)] == [1000.0, 1001.0, 1002.0]
    assert [float(dec_in[1, P + k, 0]) for k in range(2)] == [1100.0, 1101.0]
    assert stats["raw_counts"] == [0, 0]
    # M_max collapses to the chunk count of the longest sample.
    assert dec_in.shape[1] == P + 3 + 2


# ---------------------------------------------------------------------------
# Collator passthrough of the mixed extras
# ---------------------------------------------------------------------------


def test_collate_pads_mixed_extras():
    def sample(n_src, n_prompt, n_ans, doc_ids, spans):
        return {
            "source_ids": torch.arange(n_src, dtype=torch.long),
            "prompt_ids": torch.arange(n_prompt, dtype=torch.long),
            "answer_ids": torch.arange(n_ans, dtype=torch.long),
            "doc_dec_ids": torch.as_tensor(doc_ids, dtype=torch.long),
            "chunk_dec_spans": torch.as_tensor(spans, dtype=torch.long).reshape(-1, 2),
        }

    collate = make_collate_fn(dec_pad_id=99, enc_pad_id=0)
    batch = collate(
        [
            sample(4, 2, 2, [5, 6, 7], [(0, 2), (2, 3)]),
            sample(2, 3, 1, [8], [(0, 1)]),
        ]
    )
    assert batch["doc_dec_ids"].shape == (2, 3)
    assert batch["doc_dec_ids"][0].tolist() == [5, 6, 7]
    assert batch["doc_dec_ids"][1].tolist() == [8, 99, 99]  # right-pad, dec pad id
    assert batch["chunk_dec_spans"].shape == (2, 2, 2)
    assert batch["chunk_dec_spans"][1].tolist() == [[0, 1], [0, 0]]  # (0,0) span pad
    assert batch["chunk_counts"].tolist() == [2, 1]


def test_collate_without_mixed_keys_unchanged():
    def sample(n_src, n_prompt, n_ans):
        return {
            "source_ids": torch.arange(n_src, dtype=torch.long),
            "prompt_ids": torch.arange(n_prompt, dtype=torch.long),
            "answer_ids": torch.arange(n_ans, dtype=torch.long),
        }

    collate = make_collate_fn(dec_pad_id=99, enc_pad_id=0)
    batch = collate([sample(4, 2, 2), sample(2, 3, 1)])
    assert "doc_dec_ids" not in batch
    assert "chunk_dec_spans" not in batch
    assert "chunk_counts" not in batch
