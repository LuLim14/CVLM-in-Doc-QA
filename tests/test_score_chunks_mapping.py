"""Unit tests for the skim-then-read pure helpers in tools/score_chunks.py.

Stub offset mappings only — no models, no tokenizers, no network. The helpers
are imported directly; tools/score_chunks.py defers all torch/transformers
imports into main() precisely so this stays lightweight.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from score_chunks import chunk_char_spans, render_selection, select_chunks_by_budget

# Stub "document": 7 whitespace-separated pseudo-tokens, 2 chars each.
#   token:   aa   bb   cc   dd   ee   ff   gg
#   chars:  0-2  3-5  6-8  9-11 12-14 15-17 18-20
DOC = "aa bb cc dd ee ff gg"
OFFSETS = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 17), (18, 20)]


# ---------------------------------------------------------------------------
# chunk_char_spans
# ---------------------------------------------------------------------------


def test_spans_basic_cr2():
    # cr=2 over 7 tokens -> ceil(7/2)=4 chunks; last chunk is partial (1 token).
    spans = chunk_char_spans(OFFSETS, compression_rate=2, max_chunks=512)
    assert spans == [(0, 5), (6, 11), (12, 17), (18, 20)]
    assert [DOC[s:e] for s, e in spans] == ["aa bb", "cc dd", "ee ff", "gg"]


def test_spans_cr1_is_identity():
    spans = chunk_char_spans(OFFSETS, compression_rate=1, max_chunks=512)
    assert spans == [(int(s), int(e)) for s, e in OFFSETS]


def test_spans_max_chunks_cap_matches_pool_truncation():
    # _chunked_attention_pool caps V at max_vision_len and drops tail tokens;
    # the char mapping must do the same.
    spans = chunk_char_spans(OFFSETS, compression_rate=2, max_chunks=3)
    assert spans == [(0, 5), (6, 11), (12, 17)]


def test_spans_charlevel_stub_reconstructs_document():
    # Char-level "tokenizer": offsets (i, i+1). Chunk substrings must tile the
    # document exactly.
    doc = "abcdefghij"
    offsets = [(i, i + 1) for i in range(len(doc))]
    spans = chunk_char_spans(offsets, compression_rate=4, max_chunks=512)
    assert spans == [(0, 4), (4, 8), (8, 10)]
    assert "".join(doc[s:e] for s, e in spans) == doc


def test_spans_empty_offsets():
    assert chunk_char_spans([], compression_rate=4, max_chunks=512) == []


def test_spans_accepts_list_offsets():
    # Fast tokenizers may hand back lists instead of tuples (e.g. via JSON).
    spans = chunk_char_spans([[0, 2], [3, 5], [6, 8]], compression_rate=2, max_chunks=512)
    assert spans == [(0, 5), (6, 8)]


# ---------------------------------------------------------------------------
# select_chunks_by_budget
# ---------------------------------------------------------------------------


def test_select_by_importance_within_budget():
    importance = [0.1, 0.9, 0.2, 0.8]
    costs = [3, 3, 3, 3]
    chosen, cov = select_chunks_by_budget(importance, costs, budget=6)
    assert chosen == [1, 3]  # two most important, returned in document order
    assert cov == 6


def test_select_skips_unfitting_chunk_but_continues():
    # Top chunk (idx 1) fits; second-best (idx 0) would overflow and is
    # skipped; cheaper idx 2 still fits afterwards.
    importance = [0.8, 0.9, 0.5]
    costs = [5, 4, 2]
    chosen, cov = select_chunks_by_budget(importance, costs, budget=6)
    assert chosen == [1, 2]
    assert cov == 6


def test_select_never_empty_even_if_top_overshoots():
    chosen, cov = select_chunks_by_budget([0.2, 0.9], [10, 10], budget=4)
    assert chosen == [1]  # top-importance chunk taken despite overshoot
    assert cov == 10


def test_select_tie_break_prefers_document_order():
    chosen, _ = select_chunks_by_budget([0.5, 0.5, 0.5], [2, 2, 2], budget=4)
    assert chosen == [0, 1]


# ---------------------------------------------------------------------------
# render_selection
# ---------------------------------------------------------------------------


def test_render_merges_adjacent_and_separates_gaps():
    spans = chunk_char_spans(OFFSETS, compression_rate=2, max_chunks=512)
    # chunks 0,1 adjacent -> one substring incl. inter-chunk chars; chunk 3 gapped.
    assert render_selection(DOC, spans, [0, 1, 3]) == "aa bb cc dd ... gg"


def test_render_document_order_is_independent_of_importance_order():
    spans = chunk_char_spans(OFFSETS, compression_rate=2, max_chunks=512)
    imp = [0.1, 0.0, 0.9, 0.5]  # top chunks: 2, 3, 0
    costs = [5, 5, 5, 2]
    chosen, _ = select_chunks_by_budget(imp, costs, budget=12)
    assert chosen == [0, 2, 3]
    assert render_selection(DOC, spans, chosen) == "aa bb ... ee ff gg"


def test_render_single_chunk_and_empty():
    spans = chunk_char_spans(OFFSETS, compression_rate=2, max_chunks=512)
    assert render_selection(DOC, spans, [2]) == "ee ff"
    assert render_selection(DOC, spans, []) == ""
