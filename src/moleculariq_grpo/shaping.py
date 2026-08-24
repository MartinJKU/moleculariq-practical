"""Graded (partial-credit) shaping for counting rewards.

Why this exists
---------------
GRPO derives each completion's advantage from its group -- the completions
sampled for the same prompt. If every completion in a group scores identically,
``r_i - mean(r) = 0`` and the group teaches nothing. Instrumenting a real run
(``RewardGroupMonitor``) showed **38.4% of groups uniformly wrong** on counting,
with 0% uniformly correct: a third of the rollout budget produced no
correctness signal at all. In those groups the only surviving gradient came
from the format shaping term, so what GRPO actually learned there was output
formatting, not counting.

A binary reward cannot distinguish "off by one" from "off by twenty", yet those
are exactly the completions whose ordering would point the policy somewhere
useful. This module supplies that ordering:

    reward = 1.0                                if the verifier accepts
             lambda * exp(-|pred - true| / tau) if a number was parsed but wrong
             0.0                                if nothing usable was produced

Design constraints
------------------
* **Exact match always wins.** Partial credit is capped at
  ``lambda * exp(-1/tau)`` (0.107 at the defaults), far below 1.0, so no
  approximate answer can ever outrank a correct one. The policy optimum is
  unchanged -- only the gradient *around* it becomes informative.
* **Evaluation is untouched.** ``rewards.score_answer`` keeps its strictly
  binary contract, because ``scripts/evaluate.py`` uses it and reported numbers
  must stay comparable with the official protocol. Partial credit exists only
  in the training reward, and only when explicitly enabled.
* **Counting only.** Index answers are lists and constraint answers are
  molecules; neither has a meaningful scalar distance, and inventing one for
  constraint generation would risk a fresh reward-hacking surface on a task
  that already had one. Those tasks fall through to the binary reward.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Optional

from .extraction import extract_moleculariq_answer

logger = logging.getLogger(__name__)

__all__ = ["count_partial_credit", "parse_count_answer", "proximity"]

# Defaults: lambda caps partial credit well below an exact match; tau sets how
# quickly credit decays with error (at tau=3 an answer off by 3 keeps ~37% of
# the available partial credit, off by 10 keeps ~4%).
DEFAULT_LAMBDA = 0.15
DEFAULT_TAU = 3.0


def _as_dict(value: Any) -> Optional[dict]:
    """Parse a target that may arrive as a JSON string or an already-parsed dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _as_int(value: Any) -> Optional[int]:
    """Coerce a scalar answer to int, rejecting bools and non-finite floats."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        try:
            return int(text)
        except ValueError:
            try:
                f = float(text)
            except ValueError:
                return None
            return int(f) if math.isfinite(f) else None
    return None


def parse_count_answer(response_text: str, target: Any) -> Optional[dict[str, int]]:
    """Extract predicted integer counts, keyed to match the target.

    Uses the same official extraction function as the verifier, so a completion
    that the verifier cannot read scores 0 here too -- partial credit never
    rewards output the real reward would reject outright.

    Returns ``None`` when no usable number is found. When the model emits a
    bare number instead of the required JSON object and the target has exactly
    one key, that number is accepted: the format reward already covers JSON
    structure, and double-penalising it here would confound the two signals.
    """
    target_dict = _as_dict(target)
    if not target_dict:
        return None
    try:
        extracted = extract_moleculariq_answer(response_text)
    except Exception:
        logger.exception("Extraction failed during partial-credit scoring")
        return None
    if extracted is None:
        return None

    extracted_dict = _as_dict(extracted)
    if extracted_dict is not None:
        out = {}
        for key in target_dict:
            value = _as_int(extracted_dict.get(key))
            if value is not None:
                out[key] = value
        return out or None

    # Bare scalar answer: only unambiguous for a single-property question.
    if len(target_dict) == 1:
        value = _as_int(extracted)
        if value is not None:
            return {next(iter(target_dict)): value}
    return None


def proximity(predicted: int, true: int, tau: float = DEFAULT_TAU) -> float:
    """Exponential closeness in [0, 1]; 1.0 only when the values are equal."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return math.exp(-abs(predicted - true) / tau)


def count_partial_credit(response_text: str, target: Any,
                         lam: float = DEFAULT_LAMBDA,
                         tau: float = DEFAULT_TAU) -> float:
    """Partial credit in ``[0, lam)`` for a wrong-but-close count answer.

    Only called when the official verifier has already returned 0.0, so the
    answer is known to be incorrect. With several requested properties the
    per-property proximities are averaged, and a property the model omitted
    contributes 0 -- otherwise dropping the hard half of a question would raise
    its score.
    """
    if lam <= 0:
        return 0.0
    target_dict = _as_dict(target)
    if not target_dict:
        return 0.0
    predicted = parse_count_answer(response_text, target)
    if not predicted:
        return 0.0

    scores = []
    for key, true_value in target_dict.items():
        true_int = _as_int(true_value)
        if true_int is None:
            continue                       # non-numeric property: no distance
        pred_int = predicted.get(key)
        scores.append(0.0 if pred_int is None else proximity(pred_int, true_int, tau))
    if not scores:
        return 0.0
    return lam * (sum(scores) / len(scores))
