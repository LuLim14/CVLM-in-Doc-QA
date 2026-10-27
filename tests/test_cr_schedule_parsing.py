"""Unit tests for the CR schedule parser and active-cr lookup."""
import pytest

from train_cvlm import _parse_cr_schedule, _active_cr, _training_cr_for_step


def test_parse_basic():
    assert _parse_cr_schedule("1:6000,2:12000,4:18000,8:0") == [
        (1, 6000), (2, 12000), (4, 18000), (8, 0),
    ]


def test_parse_empty_returns_none():
    assert _parse_cr_schedule("") is None
    assert _parse_cr_schedule(None) is None


def test_parse_strips_whitespace():
    assert _parse_cr_schedule("  1:6000 , 2:0 ") == [(1, 6000), (2, 0)]


def test_parse_single_terminator_only():
    # "4:0" means use cr=4 forever; legal (equivalent to no schedule).
    assert _parse_cr_schedule("4:0") == [(4, 0)]


@pytest.mark.parametrize("bad", [
    "1:foo",                # non-numeric end_step
    "1,2,4,8",              # no colons
    "0:6000,8:0",           # cr=0
    "-1:6000,8:0",          # negative cr
    "1:6000,2:6000",        # last entry's end_step is not 0
    "1:6000,2:5000,4:0",    # non-last end_steps not strictly increasing
    "1:6000",               # no terminator
])
def test_parse_malformed_raises(bad):
    with pytest.raises(ValueError) as exc:
        _parse_cr_schedule(bad)
    assert str(exc.value)  # non-empty message


def test_active_cr_buckets():
    schedule = [(1, 6000), (2, 12000), (4, 18000), (8, 0)]
    assert _active_cr(schedule, 0) == 1
    assert _active_cr(schedule, 5999) == 1
    assert _active_cr(schedule, 6000) == 2
    assert _active_cr(schedule, 11999) == 2
    assert _active_cr(schedule, 12000) == 4
    assert _active_cr(schedule, 18000) == 8
    assert _active_cr(schedule, 10**9) == 8


def test_active_cr_terminator_only():
    assert _active_cr([(4, 0)], 0) == 4
    assert _active_cr([(4, 0)], 10**9) == 4


def test_training_cr_for_step_boundary_cases():
    """At end_step boundary, the saved checkpoint contains weights from the
    PRECEDING stage (the forced save fires AFTER the cr swap but the weights
    were trained at the previous cr). Hence step==end_step → previous cr."""
    schedule = [(1, 6000), (2, 12000), (4, 18000), (8, 0)]
    # Boundary: step==end_step belongs to the producing (preceding) stage.
    assert _training_cr_for_step(schedule, 6000) == 1
    assert _training_cr_for_step(schedule, 12000) == 2
    assert _training_cr_for_step(schedule, 18000) == 4
    # Mid-stage steps.
    assert _training_cr_for_step(schedule, 1) == 1
    assert _training_cr_for_step(schedule, 5999) == 1
    assert _training_cr_for_step(schedule, 7551) == 2
    assert _training_cr_for_step(schedule, 15101) == 4
    assert _training_cr_for_step(schedule, 18876) == 8
    assert _training_cr_for_step(schedule, 10**9) == 8


def test_training_cr_differs_from_active_cr_at_boundary():
    """Document the contract: at the boundary step N (end_step of stage k),
    _active_cr(N) == cr_{k+1} (next stage already started for the next step)
    while _training_cr_for_step(N) == cr_k (saved weights came from stage k)."""
    schedule = [(1, 6000), (2, 12000), (4, 18000), (8, 0)]
    assert _active_cr(schedule, 6000) == 2
    assert _training_cr_for_step(schedule, 6000) == 1
    assert _active_cr(schedule, 12000) == 4
    assert _training_cr_for_step(schedule, 12000) == 2
