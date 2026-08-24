"""Uncertainty quantification for MolecularIQ evaluation results.

Evaluation on a finite question sample is itself a measurement, so every
reported accuracy carries sampling error. This module provides the three
estimators used throughout the evaluation code:

* :func:`wilson_interval` -- confidence interval for a binomial proportion
  (pass@k, which is strictly 0/1 per question). Preferred over the normal
  approximation because it stays inside [0, 1] and remains well-behaved for
  proportions near 0 or 1 -- exactly where our small models live.
* :func:`bootstrap_ci` -- percentile bootstrap over per-question scores, used
  for ``avg_accuracy``. Unlike Wilson it makes no binomial assumption, so it
  also covers the ``--n > 1`` case where a question's score is a mean over
  several samples and therefore fractional.
* :func:`mcnemar_exact` / :func:`paired_bootstrap_diff` -- paired tests for
  "is model A actually better than model B on this eval set?". Both are
  paired: the two models answer the *same* questions, so a paired test removes
  question difficulty as a nuisance factor and is markedly more sensitive than
  comparing two independent intervals.

Only the standard library and NumPy are used, so these run anywhere the
evaluation runs (and are unit-tested in ``tests/test_stats.py``).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "wilson_interval",
    "bootstrap_ci",
    "mcnemar_exact",
    "paired_bootstrap_diff",
    "answer_diversity",
    "format_ci",
]

# Two-sided normal quantiles for the confidence levels we actually use, so the
# module needs neither SciPy nor an inverse-erf implementation.
_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def _z_for(confidence: float) -> float:
    try:
        return _Z[round(confidence, 2)]
    except KeyError as exc:
        raise ValueError(
            f"confidence must be one of {sorted(_Z)}, got {confidence}"
        ) from exc


def wilson_interval(successes: float, n: int, confidence: float = 0.95
                    ) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes: number of questions solved (may be fractional only if you have
        already rounded a mean; for fractional scores prefer :func:`bootstrap_ci`).
    n: number of questions.

    Returns ``(low, high)``, clipped to [0, 1]. Returns ``(0.0, 1.0)`` for
    ``n == 0`` -- no data means no information, not a zero-width interval.
    """
    if n <= 0:
        return 0.0, 1.0
    z = _z_for(confidence)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_ci(values: Sequence[float], confidence: float = 0.95,
                 n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-question scores.

    Resamples questions (not samples within a question) with replacement, which
    is the level the questions were drawn at and therefore the level the
    uncertainty lives at.
    """
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    if arr.size == 1 or float(arr.std()) == 0.0:
        # A degenerate sample has no spread to resample; report the point value
        # rather than a misleadingly tight interval.
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _binom_sf_le(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 0.5), computed exactly."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)


def mcnemar_exact(a_correct: Sequence[float], b_correct: Sequence[float]
                  ) -> dict:
    """Exact two-sided McNemar test on paired binary outcomes.

    Use for pass@k style metrics, where each question is solved or not by each
    model. Scores are binarized at ``> 0``. Only the *discordant* pairs carry
    information: questions both models get right (or both wrong) say nothing
    about which is better.

    Returns a dict with the discordant counts ``n_a_only`` / ``n_b_only`` and
    the exact two-sided ``p_value``.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("paired test needs equal-length score vectors "
                         f"({len(a_correct)} != {len(b_correct)})")
    a_only = sum(1 for a, b in zip(a_correct, b_correct) if a > 0 >= b)
    b_only = sum(1 for a, b in zip(a_correct, b_correct) if b > 0 >= a)
    n_disc = a_only + b_only
    # Exact binomial test: under H0 each discordant pair is a fair coin flip.
    p = min(1.0, 2.0 * _binom_sf_le(min(a_only, b_only), n_disc)) if n_disc else 1.0
    return {"n_a_only": a_only, "n_b_only": b_only, "n_discordant": n_disc,
            "p_value": p}


def paired_bootstrap_diff(a_scores: Sequence[float], b_scores: Sequence[float],
                          confidence: float = 0.95, n_boot: int = 10000,
                          seed: int = 0) -> dict:
    """Paired bootstrap CI and p-value for ``mean(a) - mean(b)``.

    Works for fractional per-question scores (``--n > 1``), where McNemar's
    binary assumption does not hold. The p-value is the two-sided bootstrap
    proportion of resampled differences that cross zero.
    """
    import numpy as np

    if len(a_scores) != len(b_scores):
        raise ValueError("paired test needs equal-length score vectors "
                         f"({len(a_scores)} != {len(b_scores)})")
    a = np.asarray(a_scores, dtype=float)
    b = np.asarray(b_scores, dtype=float)
    diff = a - b
    observed = float(diff.mean()) if diff.size else 0.0
    if diff.size == 0 or float(diff.std()) == 0.0:
        return {"diff": observed, "ci_low": observed, "ci_high": observed,
                "p_value": 1.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    # Two-sided: how often does the resampled difference fall on, or past, the
    # opposite side of zero from the observed effect?
    tail = float((boot <= 0).mean()) if observed > 0 else float((boot >= 0).mean())
    return {"diff": observed, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": min(1.0, 2.0 * tail)}


def answer_diversity(answers: Sequence[str | None]) -> dict:
    """Degeneracy statistics over a model's extracted answers.

    A model can score well on a benchmark by emitting one lucky constant
    instead of reasoning -- so the spread of answers is reported next to the
    accuracy, not left for a reader to discover. Returns the number of distinct
    answers, the most common answer and its share.
    """
    from collections import Counter

    valid = [a for a in answers if a is not None]
    if not valid:
        return {"n_answers": 0, "n_distinct": 0, "distinct_ratio": 0.0,
                "top_answer": None, "top_share": 0.0}
    counts = Counter(valid)
    top_answer, top_n = counts.most_common(1)[0]
    return {
        "n_answers": len(valid),
        "n_distinct": len(counts),
        "distinct_ratio": len(counts) / len(valid),
        "top_answer": top_answer,
        "top_share": top_n / len(valid),
    }


def format_ci(point: float, ci: tuple[float, float], digits: int = 3) -> str:
    """Render ``0.542 [0.498, 0.586]`` for logs and Markdown tables."""
    return f"{point:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"
