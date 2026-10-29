"""Unit tests for the pure helpers in tools/length_controlled_gen.py.

No models, no metrics libraries: `rouge_score` / `sacrebleu` are imported
lazily inside `Scorer.__init__`, so the sentence splitter, the protocol
transforms and the cluster-bootstrap machinery stay testable on their own.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from length_controlled_gen import (  # noqa: E402
    apply_protocol,
    bootstrap_indices,
    first_sentence,
    make_clusters,
)

# ---------------------------------------------------------------------------
# first_sentence
# ---------------------------------------------------------------------------


def test_plain_sentence_boundary():
    assert first_sentence("He was released on bail. He then left.") == "He was released on bail."


def test_question_and_exclamation():
    assert first_sentence("What happened? Nobody knows.") == "What happened?"
    assert first_sentence("Stop! Now go.") == "Stop!"


def test_abbreviation_is_not_a_boundary():
    assert first_sentence("Dr. Smith went home. Then he slept.") == "Dr. Smith went home."
    assert first_sentence("The U.S. government said yes. Then no.") == "The U.S. government said yes."


def test_initials_are_not_a_boundary():
    assert first_sentence("J. K. Rowling wrote it. She is famous.") == "J. K. Rowling wrote it."


def test_decimal_number_is_not_a_boundary():
    assert first_sentence("The price is 3.14 dollars. Really.") == "The price is 3.14 dollars."


def test_closing_quote_is_kept():
    assert first_sentence('He said "stop." Then he left.') == 'He said "stop."'


def test_lowercase_continuation_is_not_a_boundary():
    # "vs." style mid-sentence dot followed by a lowercase word
    assert first_sentence("It cost 5 dollars. then more text") == "It cost 5 dollars. then more text"


def test_newline_is_a_boundary():
    assert first_sentence("Line one\nLine two") == "Line one"
    assert first_sentence("\n\nActual answer. More.") == "Actual answer."


def test_no_boundary_returns_whole_text():
    assert first_sentence("No period here") == "No period here"


def test_empty_input():
    assert first_sentence("") == ""
    assert first_sentence("   \n  ") == ""


# ---------------------------------------------------------------------------
# apply_protocol
# ---------------------------------------------------------------------------


def test_raw_only_strips():
    assert apply_protocol("  hello world  ", "ref", "raw") == "hello world"


def test_trunc_char_uses_reference_char_length():
    pred = "abcdefghij"
    ref = "12345"
    assert apply_protocol(pred, ref, "trunc_char") == "abcde"


def test_trunc_char_is_a_noop_when_prediction_is_shorter():
    assert apply_protocol("ab", "12345", "trunc_char") == "ab"


def test_trunc_word_uses_reference_word_count():
    pred = "one two three four five"
    ref = "a b"
    assert apply_protocol(pred, ref, "trunc_word") == "one two"


def test_trunc_word_normalises_whitespace():
    assert apply_protocol("one   two\nthree", "a b c", "trunc_word") == "one two three"


def test_first_sentence_protocol_ignores_reference():
    assert apply_protocol("First. Second.", "irrelevant reference text", "first_sentence") == "First."


# ---------------------------------------------------------------------------
# cluster bootstrap
# ---------------------------------------------------------------------------


def test_make_clusters_groups_rows_by_doc_id():
    clusters = make_clusters(["a", "a", "b", "b", "b", "c"])
    assert sorted(c.tolist() for c in clusters) == [[0, 1], [2, 3, 4], [5]]


def test_bootstrap_resamples_documents_not_rows():
    # Two documents of 3 rows each: every resample must be a union of whole
    # 3-row blocks, so its size is always 6 and rows of a document co-occur.
    clusters = make_clusters(["d0"] * 3 + ["d1"] * 3)
    boots = bootstrap_indices(clusters, n_boot=50, seed=0)
    assert len(boots) == 50
    for b in boots:
        assert len(b) == 6
        counts = np.bincount(b, minlength=6)
        assert counts[0] == counts[1] == counts[2]
        assert counts[3] == counts[4] == counts[5]


def test_bootstrap_is_deterministic_for_a_seed():
    clusters = make_clusters([f"d{i//5}" for i in range(50)])
    a = bootstrap_indices(clusters, n_boot=10, seed=7)
    b = bootstrap_indices(clusters, n_boot=10, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
