"""Instrumentation for the GRPO learning signal.

GRPO computes each completion's advantage *relative to its own group* -- the
set of completions sampled for the same prompt. If every completion in a group
receives the same reward, then ``r_i - mean(r) = 0`` for all of them and the
group contributes **no gradient at all**, whatever normalisation is applied
afterwards. With a strictly binary correctness reward this happens constantly:

* an easy prompt the model always solves     -> all rewards 1.0 -> no gradient
* a hard prompt the model never solves       -> all rewards 0.0 -> no gradient

so only prompts the policy is *already borderline* on actually train it. The
fraction of degenerate groups is therefore the single most useful number for
deciding whether a run is starved of signal, and it is invisible in the usual
mean-reward curve: a run can show a healthy mean reward while most of its
groups are saturated at 1.0 and teaching nothing.

:class:`RewardGroupMonitor` wraps a TRL reward function and records that
fraction. It is a transparent pass-through -- the wrapped rewards are returned
unchanged -- so attaching it cannot alter training.

Groups are recovered from the ``prompts`` argument rather than from a
configured group size: TRL passes each prompt repeated once per generation, so
runs of an identical prompt delimit a group. That keeps the monitor correct
under gradient accumulation and uneven final batches, where a fixed reshape
would silently mis-group.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = ["RewardGroupMonitor", "group_by_prompt", "summarize_groups"]


def _prompt_key(prompt: Any) -> str:
    """A hashable identity for a prompt in either chat or plain-text form."""
    if isinstance(prompt, str):
        return prompt
    try:
        return json.dumps(prompt, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(prompt)


def group_by_prompt(prompts: Optional[list], rewards: list[float]
                    ) -> list[list[float]]:
    """Split a flat reward list into per-prompt groups.

    Consecutive identical prompts form one group. With no prompts available the
    whole batch is treated as a single group, which is the conservative choice:
    it can only understate degeneracy, never invent it.
    """
    if not prompts or len(prompts) != len(rewards):
        return [list(rewards)]
    groups: list[list[float]] = []
    current: list[float] = []
    previous: Optional[str] = None
    for prompt, reward in zip(prompts, rewards):
        key = _prompt_key(prompt)
        if previous is not None and key != previous:
            groups.append(current)
            current = []
        current.append(reward)
        previous = key
    if current:
        groups.append(current)
    return groups


def summarize_groups(groups: list[list[float]]) -> dict:
    """Degeneracy statistics over one batch of groups.

    ``frac_no_signal`` is the headline: the share of groups that produce zero
    gradient. It is split into ``frac_all_zero`` (too hard -- the model never
    succeeds) and ``frac_all_max`` (too easy -- already saturated), because the
    two call for opposite fixes: a denser reward for the former, harder prompts
    for the latter.
    """
    usable = [g for g in groups if len(g) > 1]
    if not usable:
        return {"n_groups": len(groups), "frac_no_signal": float("nan"),
                "frac_all_zero": float("nan"), "frac_all_max": float("nan"),
                "mean_group_std": float("nan"), "mean_reward": float("nan")}

    n = len(usable)
    n_flat = n_zero = n_max = 0
    std_total = 0.0
    reward_total = 0.0
    for g in usable:
        lo, hi = min(g), max(g)
        mean = sum(g) / len(g)
        reward_total += mean
        var = sum((r - mean) ** 2 for r in g) / len(g)
        std_total += var ** 0.5
        if hi - lo < 1e-12:          # every completion scored identically
            n_flat += 1
            if hi <= 1e-12:
                n_zero += 1
            else:
                n_max += 1
    return {
        "n_groups": n,
        "frac_no_signal": n_flat / n,
        "frac_all_zero": n_zero / n,
        "frac_all_max": n_max / n,
        "mean_group_std": std_total / n,
        "mean_reward": reward_total / n,
    }


class RewardGroupMonitor:
    """Pass-through wrapper around a TRL reward function that logs degeneracy.

    Usage in a trainer setup::

        monitor = RewardGroupMonitor(correctness_reward, log_every=10,
                                     out_path="outputs/run/reward_groups.jsonl")
        trainer = GRPOTrainer(reward_funcs=[monitor], ...)

    The wrapper copies ``__name__`` from the wrapped function so TRL's own
    per-reward logging keeps its usual column name.
    """

    def __init__(self, reward_func: Callable, log_every: int = 10,
                 out_path: Optional[str | Path] = None):
        self.reward_func = reward_func
        self.log_every = max(1, int(log_every))
        self.out_path = Path(out_path) if out_path else None
        self.calls = 0
        self._totals: Counter = Counter()
        self._batches = 0
        self._last_by_task: dict = {}
        # TRL identifies reward functions by __name__ for its logged metrics.
        self.__name__ = getattr(reward_func, "__name__", "reward")
        if self.out_path:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *args, **kwargs) -> list[float]:
        rewards = self.reward_func(*args, **kwargs)
        try:
            prompts = kwargs.get("prompts")
            groups = group_by_prompt(prompts, rewards)
            stats = summarize_groups(groups)
            # In a multitask run the aggregate hides the case that matters: one
            # task starved of signal while the others train normally. Break the
            # statistics down per task whenever task labels are available.
            task_type = kwargs.get("task_type")
            if task_type:
                task_groups = group_by_prompt(prompts, list(task_type))
                if len(task_groups) == len(groups):
                    by_task: dict[str, list[list[float]]] = {}
                    for g, tg in zip(groups, task_groups):
                        by_task.setdefault(str(tg[0]) if tg else "unknown", []).append(g)
                    stats["by_task"] = {t: summarize_groups(gs)
                                        for t, gs in sorted(by_task.items())}
            self._record(stats)
        except Exception:
            # Instrumentation must never take down a training run.
            logger.exception("Reward group monitoring failed (training continues)")
        return rewards

    def _record(self, stats: dict) -> None:
        self.calls += 1
        self._last_by_task = stats.get("by_task") or {}
        if stats["n_groups"] and stats["frac_no_signal"] == stats["frac_no_signal"]:
            self._batches += 1
            for key in ("frac_no_signal", "frac_all_zero", "frac_all_max",
                        "mean_group_std", "mean_reward"):
                self._totals[key] += stats[key]
        if self.out_path:
            with open(self.out_path, "a") as f:
                f.write(json.dumps({"call": self.calls, **stats}) + "\n")
        if self.calls % self.log_every == 0 and self._batches:
            avg = {k: v / self._batches for k, v in self._totals.items()}
            logger.info(
                "[reward groups] call %d | no-signal %.1f%% "
                "(all-wrong %.1f%%, all-correct %.1f%%) | "
                "mean group std %.3f | mean reward %.3f",
                self.calls, 100 * avg["frac_no_signal"],
                100 * avg["frac_all_zero"], 100 * avg["frac_all_max"],
                avg["mean_group_std"], avg["mean_reward"])
            if self._last_by_task:
                parts = [f"{t}: no-signal {100*s['frac_no_signal']:.0f}% "
                         f"reward {s['mean_reward']:.3f}"
                         for t, s in sorted(self._last_by_task.items())]
                logger.info("[reward groups]   per task | %s", "  |  ".join(parts))
            if avg["frac_no_signal"] > 0.5:
                logger.warning(
                    "[reward groups] over half of all groups produce no gradient "
                    "(%.0f%% all-wrong, %.0f%% all-correct). A binary reward is "
                    "starving this run; consider a graded reward that orders "
                    "incorrect answers, or a difficulty-matched prompt mix.",
                    100 * avg["frac_all_zero"], 100 * avg["frac_all_max"])
