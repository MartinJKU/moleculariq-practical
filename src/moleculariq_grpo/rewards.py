"""Reward functions for GRPO training, backed by the MolecularIQ verifiers.

The single source of truth for correctness is ``moleculariq_core.evaluate_answer``
(the same dispatcher the official evaluation harness uses), fed with answers
extracted by the official extraction function. Rewards are binary per sample:
1.0 if the answer is verified correct (all requested properties / all
constraints), else 0.0.

Both functions follow the TRL ``GRPOTrainer`` reward contract: they are called
with ``prompts``, ``completions``, ``completion_ids`` and every extra dataset
column as keyword lists (here: ``task_type``, ``target``, ``constraints``), and
return a list of floats.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from moleculariq_core import evaluate_answer

from .extraction import extract_moleculariq_answer

logger = logging.getLogger(__name__)

# Safety cap: never feed absurdly long extracted answers into RDKit.
_MAX_EXTRACTED_LEN = 4096


def _completion_text(completion: Any) -> str:
    """Get the raw text out of a TRL completion (conversational or plain)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        # Conversational format: [{"role": "assistant", "content": "..."}]
        last = completion[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            if isinstance(content, str):
                return content
    return str(completion)


def _normalize_task_type(task_type: str) -> str:
    """Map dataset task types onto ``evaluate_answer`` task types.

    Mirrors the official ``task_processor.moleculariq_bencheval``: the
    benchmark stores ``count`` / ``index`` / ``generation`` (and the
    single_/multi_ variants map onto the same reward functions).
    """
    tt = (task_type or "").lower()
    if "constraint" in tt or "generation" in tt:
        return "constraint_generation"
    return tt


def score_answer(
    response_text: str,
    task_type: str,
    target: Optional[str] = None,
    constraints: Optional[str] = None,
) -> float:
    """Extract and verify a single model response. Returns 1.0 or 0.0.

    ``target`` and ``constraints`` may be JSON strings (as stored in the
    datasets) — the moleculariq_core reward functions parse them internally.
    """
    try:
        extracted = extract_moleculariq_answer(response_text)
        if extracted is None:
            return 0.0
        if isinstance(extracted, str):
            if not extracted.strip():
                return 0.0
            extracted = extracted[:_MAX_EXTRACTED_LEN]

        tt = _normalize_task_type(task_type)
        if tt == "constraint_generation":
            if constraints is None:
                return 0.0
            reward = evaluate_answer(
                task_type=tt,
                predicted=extracted,
                constraints=constraints,
                return_details=False,
            )
        else:
            if target is None:
                return 0.0
            reward = evaluate_answer(
                task_type=tt,
                predicted=extracted,
                target=target,
                return_details=False,
            )
        if isinstance(reward, dict):
            reward = reward.get("reward", 0.0)
        return float(reward)
    except Exception:
        logger.exception("Reward computation failed; assigning 0.0")
        return 0.0


def correctness_reward(
    prompts=None,
    completions=None,
    completion_ids=None,
    task_type=None,
    target=None,
    constraints=None,
    **kwargs,
) -> list[float]:
    """Binary MolecularIQ verification reward (the actual training signal)."""
    n = len(completions)
    task_type = task_type if task_type is not None else [None] * n
    target = target if target is not None else [None] * n
    constraints = constraints if constraints is not None else [None] * n

    rewards = []
    for completion, tt, tgt, cons in zip(completions, task_type, target, constraints):
        text = _completion_text(completion)
        rewards.append(score_answer(text, tt, target=tgt, constraints=cons))
    return rewards


def format_reward(
    prompts=None,
    completions=None,
    completion_ids=None,
    **kwargs,
) -> list[float]:
    """Small shaping reward for producing ``<answer>{valid JSON object}</answer>``.

    Helps a small model discover the required output format early in training.
    Weighted low (see ``reward_weights`` in the training config) so it cannot
    dominate the correctness signal.
    """
    rewards = []
    for completion in completions:
        text = _completion_text(completion)
        score = 0.0
        if "<answer>" in text and "</answer>" in text:
            inner = text.split("<answer>")[-1].split("</answer>")[0].strip()
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict) and parsed:
                    score = 1.0
            except (json.JSONDecodeError, ValueError):
                score = 0.0
        rewards.append(score)
    return rewards
