"""Unit tests for moleculariq_grpo.stats.

Values are checked against closed-form results or published worked examples
rather than against the implementation's own output, so a regression in the
formulas is actually caught.

Run with:  pytest -q
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moleculariq_grpo.stats import (  # noqa: E402
    answer_diversity,
    bootstrap_ci,
    format_ci,
    mcnemar_exact,
    paired_bootstrap_diff,
    wilson_interval,
)


# --------------------------------------------------------------------------
# Wilson interval
# --------------------------------------------------------------------------

def test_wilson_matches_closed_form():
    """Textbook example: ic=0.95, 10 successes out of 20 -> [0.299, 0.701]."""
    lo, hi = wilson_interval(10, 20, 0.95)
    assert lo == pytest.approx(0.2989, abs=1e-3)
    assert hi == pytest.approx(0.7011, abs=1e-3)


def test_wilson_is_centred_for_half():
    lo, hi = wilson_interval(50, 100)
    assert (lo + hi) / 2 == pytest.approx(0.5, abs=1e-9)


def test_wilson_stays_in_unit_interval_at_extremes():
    """The normal approximation would leave [0,1] here; Wilson must not."""
    for successes, n in ((0, 10), (10, 10), (0, 1), (1, 1)):
        lo, hi = wilson_interval(successes, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_narrows_as_n_grows():
    widths = [wilson_interval(n // 2, n)[1] - wilson_interval(n // 2, n)[0]
              for n in (10, 100, 1000, 10000)]
    assert widths == sorted(widths, reverse=True)
    # Width should shrink roughly like 1/sqrt(n).
    assert widths[-1] < widths[0] / 10


def test_wilson_no_data_is_maximally_uncertain():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_higher_confidence_is_wider():
    w90 = wilson_interval(30, 100, 0.90)
    w99 = wilson_interval(30, 100, 0.99)
    assert (w99[1] - w99[0]) > (w90[1] - w90[0])


def test_wilson_rejects_unsupported_confidence():
    with pytest.raises(ValueError):
        wilson_interval(5, 10, 0.42)


# --------------------------------------------------------------------------
# Bootstrap CI
# --------------------------------------------------------------------------

def test_bootstrap_brackets_the_mean():
    values = [0.0] * 60 + [1.0] * 40
    lo, hi = bootstrap_ci(values, seed=0)
    assert lo < 0.40 < hi


def test_bootstrap_is_deterministic_given_seed():
    values = [0.0, 1.0] * 50
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_bootstrap_agrees_with_wilson_on_binary_data():
    """Two valid estimators of the same quantity should broadly concur."""
    values = [1.0] * 25 + [0.0] * 75
    b_lo, b_hi = bootstrap_ci(values, seed=0)
    w_lo, w_hi = wilson_interval(25, 100)
    assert b_lo == pytest.approx(w_lo, abs=0.05)
    assert b_hi == pytest.approx(w_hi, abs=0.05)


def test_bootstrap_handles_fractional_scores():
    """--n > 1 gives per-question means in [0,1]; these must be supported."""
    lo, hi = bootstrap_ci([0.33, 0.66, 1.0, 0.0, 0.5] * 20, seed=1)
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_degenerate_inputs():
    assert bootstrap_ci([]) == (0.0, 1.0)
    assert bootstrap_ci([0.7] * 10) == (0.7, 0.7)


# --------------------------------------------------------------------------
# McNemar
# --------------------------------------------------------------------------

def test_mcnemar_identical_models_not_significant():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0] * 10
    res = mcnemar_exact(scores, scores)
    assert res["n_discordant"] == 0
    assert res["p_value"] == 1.0


def test_mcnemar_counts_only_discordant_pairs():
    a = [1.0, 1.0, 0.0, 0.0]
    b = [1.0, 0.0, 1.0, 0.0]
    res = mcnemar_exact(a, b)
    assert res["n_a_only"] == 1     # question 2
    assert res["n_b_only"] == 1     # question 3
    assert res["n_discordant"] == 2


def test_mcnemar_exact_p_value_matches_binomial():
    """10 discordant pairs, all favouring A: p = 2 * 0.5^10."""
    a = [1.0] * 10 + [0.0] * 5
    b = [0.0] * 10 + [0.0] * 5
    res = mcnemar_exact(a, b)
    assert res["n_a_only"] == 10 and res["n_b_only"] == 0
    assert res["p_value"] == pytest.approx(2 * 0.5 ** 10, rel=1e-9)


def test_mcnemar_detects_a_clear_win():
    a = [1.0] * 40 + [0.0] * 60
    b = [0.0] * 100
    assert mcnemar_exact(a, b)["p_value"] < 0.001


def test_mcnemar_is_symmetric_in_p():
    a = [1.0] * 12 + [0.0] * 8
    b = [0.0] * 12 + [1.0] * 8
    assert mcnemar_exact(a, b)["p_value"] == pytest.approx(
        mcnemar_exact(b, a)["p_value"])


def test_mcnemar_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar_exact([1.0, 0.0], [1.0])


# --------------------------------------------------------------------------
# Paired bootstrap
# --------------------------------------------------------------------------

def test_paired_bootstrap_reports_observed_difference():
    a = [1.0] * 50 + [0.0] * 50
    b = [0.0] * 100
    res = paired_bootstrap_diff(a, b, seed=0)
    assert res["diff"] == pytest.approx(0.5)
    assert res["ci_low"] > 0.0
    assert res["p_value"] < 0.05


def test_paired_bootstrap_null_effect_is_not_significant():
    a = [1.0, 0.0] * 50
    b = [1.0, 0.0] * 50
    res = paired_bootstrap_diff(a, b, seed=0)
    assert res["diff"] == pytest.approx(0.0)
    assert res["p_value"] == 1.0


def test_paired_bootstrap_ci_brackets_diff():
    a = [0.8, 0.6, 1.0, 0.4] * 25
    b = [0.5, 0.5, 0.5, 0.5] * 25
    res = paired_bootstrap_diff(a, b, seed=3)
    assert res["ci_low"] <= res["diff"] <= res["ci_high"]


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0], [1.0, 0.0])


# --------------------------------------------------------------------------
# Answer diversity (mode-collapse detection)
# --------------------------------------------------------------------------

def test_answer_diversity_flags_a_constant_policy():
    res = answer_diversity(["CC=O"] * 100)
    assert res["n_distinct"] == 1
    assert res["top_share"] == 1.0
    assert res["top_answer"] == "CC=O"


def test_answer_diversity_on_fully_distinct_answers():
    res = answer_diversity([f"C{i}" for i in range(50)])
    assert res["n_distinct"] == 50
    assert res["distinct_ratio"] == 1.0
    assert res["top_share"] == pytest.approx(1 / 50)


def test_answer_diversity_ignores_none():
    res = answer_diversity(["A", None, "A", None, "B"])
    assert res["n_answers"] == 3
    assert res["n_distinct"] == 2
    assert res["top_share"] == pytest.approx(2 / 3)


def test_answer_diversity_empty():
    res = answer_diversity([None, None])
    assert res["n_answers"] == 0
    assert res["top_answer"] is None


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def test_format_ci():
    assert format_ci(0.5421, (0.4981, 0.5862)) == "0.542 [0.498, 0.586]"
