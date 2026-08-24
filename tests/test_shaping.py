"""Unit tests for moleculariq_grpo.shaping (graded count reward).

The safety property under test is that partial credit can never outrank an
exact answer -- if it could, the policy optimum would move and the graded
reward would be training the model to be approximately right on purpose.

Run with:  pytest -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moleculariq_grpo.shaping import (  # noqa: E402
    DEFAULT_LAMBDA,
    DEFAULT_TAU,
    count_partial_credit,
    parse_count_answer,
    proximity,
)


def answer(payload: str) -> str:
    """Wrap a payload the way the system prompt requires."""
    return f"Some reasoning here.\n<answer>{payload}</answer>"


# --------------------------------------------------------------------------
# The core safety property
# --------------------------------------------------------------------------

def test_partial_credit_never_reaches_an_exact_match():
    """No approximate answer may score at or above the exact-match reward 1.0."""
    target = '{"ring_count": 5}'
    for pred in range(-50, 60):
        credit = count_partial_credit(answer(f'{{"ring_count": {pred}}}'), target)
        assert credit < 1.0
        if pred != 5:
            # Off-by-one is the best possible near miss; it must stay far below 1.
            assert credit <= DEFAULT_LAMBDA


def test_closest_wrong_answer_is_bounded_by_lambda_exp():
    """The maximum achievable partial credit is lam * exp(-1/tau)."""
    got = count_partial_credit(answer('{"ring_count": 6}'), '{"ring_count": 5}')
    import math
    assert got == pytest.approx(DEFAULT_LAMBDA * math.exp(-1 / DEFAULT_TAU))
    assert got < 0.11


def test_credit_is_monotonically_decreasing_in_error():
    target = '{"ring_count": 10}'
    credits = [count_partial_credit(answer(f'{{"ring_count": {10 + e}}}'), target)
               for e in range(1, 12)]
    assert credits == sorted(credits, reverse=True)
    assert all(c > 0 for c in credits)


def test_far_answers_earn_almost_nothing():
    c = count_partial_credit(answer('{"ring_count": 90}'), '{"ring_count": 2}')
    assert c < 1e-6


# --------------------------------------------------------------------------
# proximity
# --------------------------------------------------------------------------

def test_proximity_is_one_only_when_equal():
    assert proximity(7, 7) == 1.0
    assert proximity(7, 8) < 1.0


def test_proximity_is_symmetric():
    assert proximity(3, 9) == pytest.approx(proximity(9, 3))


def test_proximity_rejects_nonpositive_tau():
    with pytest.raises(ValueError):
        proximity(1, 2, tau=0)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_parses_json_answer():
    assert parse_count_answer(answer('{"ring_count": 4}'),
                              '{"ring_count": 9}') == {"ring_count": 4}


def test_parses_bare_number_for_single_property():
    """The format reward already covers JSON structure; don't double-penalise."""
    assert parse_count_answer(answer("4"), '{"ring_count": 9}') == {"ring_count": 4}


def test_bare_number_rejected_when_ambiguous():
    """With two requested properties a lone number cannot be assigned."""
    assert parse_count_answer(answer("4"),
                              '{"ring_count": 9, "carbon_count": 3}') is None


def test_parse_accepts_target_as_dict():
    assert parse_count_answer(answer('{"c": 2}'), {"c": 5}) == {"c": 2}


def test_parse_returns_none_for_unusable_output():
    for text in ("", "I don't know.", answer("not a number")):
        assert parse_count_answer(text, '{"ring_count": 3}') is None


def test_parse_ignores_keys_not_in_target():
    got = parse_count_answer(answer('{"wrong_key": 4}'), '{"ring_count": 9}')
    assert got is None


def test_parse_handles_malformed_target():
    assert parse_count_answer(answer('{"c": 1}'), "not json") is None
    assert parse_count_answer(answer('{"c": 1}'), None) is None


# --------------------------------------------------------------------------
# Credit computation
# --------------------------------------------------------------------------

def test_unparseable_output_earns_zero():
    assert count_partial_credit("no answer at all", '{"ring_count": 3}') == 0.0


def test_zero_lambda_disables_shaping():
    assert count_partial_credit(answer('{"ring_count": 4}'),
                                '{"ring_count": 5}', lam=0.0) == 0.0


def test_multi_property_averages_proximities():
    """Two properties, one exact and one off by a lot -> mid-range credit."""
    target = '{"a": 5, "b": 5}'
    both_close = count_partial_credit(answer('{"a": 5, "b": 5}'), target)
    one_close = count_partial_credit(answer('{"a": 5, "b": 40}'), target)
    assert both_close > one_close > 0


def test_omitted_property_scores_zero_not_dropped():
    """Answering only the easy half must not beat answering both."""
    target = '{"a": 5, "b": 5}'
    partial = count_partial_credit(answer('{"a": 5}'), target)
    complete = count_partial_credit(answer('{"a": 5, "b": 6}'), target)
    assert complete > partial


def test_non_numeric_target_values_are_skipped():
    got = count_partial_credit(answer('{"formula": "C6H6"}'),
                               '{"formula": "C6H12"}')
    assert got == 0.0


def test_credit_scales_with_lambda():
    args = (answer('{"c": 4}'), '{"c": 5}')
    assert count_partial_credit(*args, lam=0.30) == pytest.approx(
        2 * count_partial_credit(*args, lam=0.15))


def test_tau_controls_decay_sharpness():
    args = (answer('{"c": 15}'), '{"c": 10}')
    assert count_partial_credit(*args, tau=10.0) > count_partial_credit(*args, tau=1.0)


def test_booleans_are_not_treated_as_integers():
    assert parse_count_answer(answer('{"c": true}'), '{"c": 1}') is None
