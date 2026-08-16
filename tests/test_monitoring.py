"""Unit tests for moleculariq_grpo.monitoring.

The grouping logic is the part that could silently be wrong (and would then
report reassuring numbers), so it is tested against hand-constructed batches
whose correct grouping is obvious by inspection.

Run with:  pytest -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moleculariq_grpo.monitoring import (  # noqa: E402
    RewardGroupMonitor,
    group_by_prompt,
    summarize_groups,
)


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------

def test_groups_consecutive_identical_prompts():
    prompts = ["a", "a", "a", "b", "b", "b"]
    rewards = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert group_by_prompt(prompts, rewards) == [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]


def test_groups_handle_uneven_sizes():
    """Gradient accumulation and final batches can leave uneven groups."""
    prompts = ["a", "a", "b", "b", "b", "c"]
    rewards = [1.0, 0.0, 0.0, 1.0, 1.0, 0.5]
    assert group_by_prompt(prompts, rewards) == [[1.0, 0.0], [0.0, 1.0, 1.0], [0.5]]


def test_groups_chat_format_prompts():
    p = [{"role": "user", "content": "q1"}]
    q = [{"role": "user", "content": "q2"}]
    groups = group_by_prompt([p, p, q, q], [1.0, 0.0, 0.0, 0.0])
    assert groups == [[1.0, 0.0], [0.0, 0.0]]


def test_missing_prompts_falls_back_to_one_group():
    """Conservative fallback: understate degeneracy rather than invent it."""
    assert group_by_prompt(None, [1.0, 0.0]) == [[1.0, 0.0]]
    assert group_by_prompt([], [1.0, 0.0]) == [[1.0, 0.0]]


def test_mismatched_lengths_falls_back():
    assert group_by_prompt(["a"], [1.0, 0.0]) == [[1.0, 0.0]]


def test_repeated_prompt_later_in_batch_is_a_separate_group():
    """Non-adjacent repeats are distinct groups; TRL emits them contiguously."""
    groups = group_by_prompt(["a", "a", "b", "b", "a", "a"], [1, 1, 0, 0, 0, 0])
    assert len(groups) == 3


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------

def test_all_wrong_group_has_no_signal():
    s = summarize_groups([[0.0, 0.0, 0.0, 0.0]])
    assert s["frac_no_signal"] == 1.0
    assert s["frac_all_zero"] == 1.0
    assert s["frac_all_max"] == 0.0
    assert s["mean_group_std"] == 0.0


def test_all_correct_group_has_no_signal():
    s = summarize_groups([[1.0, 1.0, 1.0]])
    assert s["frac_no_signal"] == 1.0
    assert s["frac_all_zero"] == 0.0
    assert s["frac_all_max"] == 1.0


def test_mixed_group_has_signal():
    s = summarize_groups([[1.0, 0.0, 1.0, 0.0]])
    assert s["frac_no_signal"] == 0.0
    assert s["mean_group_std"] == pytest.approx(0.5)


def test_fractions_across_mixed_batch():
    groups = [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
    s = summarize_groups(groups)
    assert s["n_groups"] == 4
    assert s["frac_no_signal"] == pytest.approx(0.75)
    assert s["frac_all_zero"] == pytest.approx(0.5)
    assert s["frac_all_max"] == pytest.approx(0.25)


def test_graded_rewards_break_a_degenerate_group():
    """The motivating case: a graded reward rescues an all-incorrect group."""
    binary = summarize_groups([[0.0, 0.0, 0.0, 0.0]])
    graded = summarize_groups([[0.10, 0.02, 0.05, 0.01]])
    assert binary["frac_no_signal"] == 1.0
    assert graded["frac_no_signal"] == 0.0
    assert graded["mean_group_std"] > 0


def test_singleton_groups_are_excluded():
    """A group of one has no within-group comparison to make."""
    s = summarize_groups([[1.0], [0.0]])
    assert s["n_groups"] == 2
    assert s["frac_no_signal"] != s["frac_no_signal"]  # NaN


def test_empty_input():
    s = summarize_groups([])
    assert s["n_groups"] == 0


# --------------------------------------------------------------------------
# Wrapper behaviour
# --------------------------------------------------------------------------

def test_monitor_is_transparent():
    """Rewards must pass through byte-for-byte; instrumentation cannot alter training."""
    def rf(prompts=None, completions=None, **kw):
        return [1.0, 0.0, 1.0, 0.0]

    m = RewardGroupMonitor(rf, log_every=1000)
    out = m(prompts=["a", "a", "b", "b"], completions=[1, 2, 3, 4])
    assert out == [1.0, 0.0, 1.0, 0.0]


def test_monitor_preserves_name_for_trl_logging():
    def correctness_reward(**kw):
        return [1.0]
    assert RewardGroupMonitor(correctness_reward).__name__ == "correctness_reward"


def test_monitor_survives_a_broken_batch():
    """Instrumentation must never take down a training run."""
    def rf(**kw):
        return [1.0, 0.0]

    m = RewardGroupMonitor(rf, log_every=1000)
    # Unhashable / odd prompt objects must not raise.
    assert m(prompts=[{1, 2}, {3, 4}], completions=[1, 2]) == [1.0, 0.0]


def test_monitor_writes_jsonl(tmp_path):
    import json

    def rf(**kw):
        return [0.0, 0.0, 1.0, 0.0]

    out = tmp_path / "sub" / "reward_groups.jsonl"
    m = RewardGroupMonitor(rf, log_every=1000, out_path=out)
    m(prompts=["a", "a", "b", "b"], completions=[1, 2, 3, 4])
    m(prompts=["a", "a", "b", "b"], completions=[1, 2, 3, 4])
    rows = [json.loads(l) for l in out.open()]
    assert len(rows) == 2
    assert rows[0]["n_groups"] == 2
    assert rows[1]["call"] == 2
