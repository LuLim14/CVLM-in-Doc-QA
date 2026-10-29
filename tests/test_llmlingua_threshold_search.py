import numpy as np

from tools.llmlingua_select import exact_threshold_selection, threshold_q_candidates


def _keep_signature(weighted_probs, q):
    threshold = np.percentile(weighted_probs, q)
    return tuple(p > threshold or (threshold == 1.0 and p == threshold) for p in weighted_probs)


def test_threshold_candidates_cover_every_dense_grid_keep_set():
    weighted = [0.05, 0.2, 0.2, 0.6, 0.9, 0.9]
    scored = [([str(i) for i in range(len(weighted))], weighted, weighted)]
    candidate_sets = {_keep_signature(weighted, q) for q in threshold_q_candidates(scored)}
    dense_sets = {_keep_signature(weighted, q) for q in np.linspace(0.0, 100.0, 10001)}
    assert dense_sets <= candidate_sets


def test_exact_selection_handles_nonmonotone_decoder_lengths():
    weighted = [0.1, 0.2, 0.3, 0.4]
    scored = [(["a", "b", "c", "d"], weighted, weighted)]

    def render(_scored, q):
        return "four" if q < 25 else "three" if q < 50 else "two" if q < 75 else "one"

    # The decoder length rises after dropping a word. The lowest feasible q
    # is still selected instead of assuming monotonic token length.
    lengths = {"four": 8, "three": 5, "two": 6, "one": 2}
    text, tokens, q = exact_threshold_selection(scored, render, lengths.__getitem__, budget=5)
    assert (text, tokens) == ("three", 5)
    assert 25 <= q < 50
