"""Question dataset generation from the MolecularIQ training molecule pool.

Molecules come from ``ml-jku/moleculariq-trainPool`` (1.27M SMILES with
complexity bins). Questions are generated with ``moleculariq_core.MolecularIQD``
— the same machinery behind the official benchmark — and written in the same
schema as the official benchmark dataset (``ml-jku/moleculariq-v0.0``):

    uid, task_type, features, question, target, constraints, original_smiles,
    complexity_bin, multi_task_load, metadata

so that one reward/eval code path works for generated and official data alike.

Task types
----------
- ``count``      -> single_count / multi_count questions   (task_type "count")
- ``index``      -> single_index / multi_index questions   (task_type "index")
- ``constraint`` -> constrained generation questions       (task_type "generation")

For ``constraint`` tasks, constraints are *anchored* on a real pool molecule:
we compute the anchor's property value and construct constraints the anchor
itself satisfies, then double-check with the official verifier
(``validate_constraint_answer``). Every emitted question is therefore
guaranteed to be satisfiable and verifier-supported.
"""
from __future__ import annotations

import itertools
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from moleculariq_core import (
    CONSTRAINT_MAP,
    COUNT_MAP,
    INDEX_MAP,
    MolecularIQD,
)

logger = logging.getLogger(__name__)

TASK_COUNT = "count"
TASK_INDEX = "index"
TASK_CONSTRAINT = "constraint"
TASKS = (TASK_COUNT, TASK_INDEX, TASK_CONSTRAINT)

# Dataset task_type field values (official benchmark convention)
_TASK_TYPE_FIELD = {TASK_COUNT: "count", TASK_INDEX: "index", TASK_CONSTRAINT: "generation"}


@dataclass
class GenerationConfig:
    task: str = TASK_COUNT
    constructs: Optional[list[str]] = None   # None = all available constructs
    num_items: int = 1                       # properties (count/index) or constraints per question
    seed: int = 42
    max_zero_fraction: float = 0.25          # cap on all-zero/all-empty answers (count/index)
    max_smiles_len: int = 200                # char cap on question SMILES (keeps prompts short)
    complexity_bins: tuple[str, ...] = ("0-250", "250-1000", "1000+")
    max_attempts_factor: int = 60            # give up after n_questions * factor attempts


# ---------------------------------------------------------------------------
# Construct map
# ---------------------------------------------------------------------------

def functional_group_names(mqd: MolecularIQD) -> list[str]:
    """Derive the functional group vocabulary from the solver itself."""
    probe = mqd.solver.functional_group_solver.get_counts_and_indices("CCO")
    names = sorted({
        k[len("functional_group_"):-len("_count")]
        for k in probe
        if k.startswith("functional_group_") and k.endswith("_count")
    })
    return names


def build_construct_map(task: str, mqd: MolecularIQD) -> dict[str, list[str]]:
    """Map construct name -> candidate property names for the given task.

    Constructs are the keys of the official COUNT_MAP / INDEX_MAP /
    CONSTRAINT_MAP (e.g. "aromatic_ring", "hba", "oxidation_state"), plus
    "functional_group" whose properties are derived from the SMARTS library.
    """
    base = {
        TASK_COUNT: COUNT_MAP,
        TASK_INDEX: INDEX_MAP,
        TASK_CONSTRAINT: CONSTRAINT_MAP,
    }[task]

    cmap = {k: list(v) for k, v in base.items() if v and k != "functional_groups"}

    fg_names = functional_group_names(mqd)
    if task == TASK_INDEX:
        fg_props = [f"functional_group_{g}_index" for g in fg_names]
    else:
        fg_props = [f"functional_group_{g}_count" for g in fg_names]
    cmap["functional_group"] = fg_props
    return cmap


def available_constructs(task: str) -> list[str]:
    mqd = MolecularIQD(seed=0)
    return sorted(build_construct_map(task, mqd).keys())


# ---------------------------------------------------------------------------
# Molecule sampling
# ---------------------------------------------------------------------------

@dataclass
class _BinSampler:
    """Round-robin over complexity bins, without-replacement inside each bin."""
    rng: random.Random
    by_bin: dict[str, list[dict]] = field(default_factory=dict)

    def __post_init__(self):
        self._cursors = {b: 0 for b in self.by_bin}
        for mols in self.by_bin.values():
            self.rng.shuffle(mols)
        self._bin_cycle = itertools.cycle(sorted(self.by_bin.keys()))

    def next_molecule(self) -> dict:
        for _ in range(len(self.by_bin)):
            b = next(self._bin_cycle)
            mols = self.by_bin[b]
            if not mols:
                continue
            i = self._cursors[b]
            if i >= len(mols):  # exhausted: reshuffle and restart
                self.rng.shuffle(mols)
                self._cursors[b] = 0
                i = 0
            self._cursors[b] = i + 1
            return mols[i]
        raise RuntimeError("No molecules available in any complexity bin")


def split_pool(
    pool_rows: list[dict],
    cfg: GenerationConfig,
    val_reserve: int = 5000,
) -> tuple[list[dict], list[dict]]:
    """Filter the molecule pool and split into disjoint train/val molecule sets."""
    rng = random.Random(cfg.seed)
    filtered = [
        r for r in pool_rows
        if r["complexity_bin"] in cfg.complexity_bins
        and len(r["smiles"]) <= cfg.max_smiles_len
    ]
    if not filtered:
        raise ValueError(
            "No pool molecules left after filtering "
            f"(bins={cfg.complexity_bins}, max_smiles_len={cfg.max_smiles_len})"
        )
    rng.shuffle(filtered)
    val_reserve = min(val_reserve, max(1, len(filtered) // 10))
    return filtered[val_reserve:], filtered[:val_reserve]


def _group_by_bin(rows: list[dict]) -> dict[str, list[dict]]:
    by_bin: dict[str, list[dict]] = {}
    for r in rows:
        by_bin.setdefault(r["complexity_bin"], []).append(dict(r))
    return by_bin


# ---------------------------------------------------------------------------
# Count / index question generation
# ---------------------------------------------------------------------------

def _is_zero_answer(target: dict) -> bool:
    """True if every requested property is absent (0 count / empty index list)."""
    for v in target.values():
        if isinstance(v, list):
            if v:
                return False
        elif isinstance(v, (int, float)):
            if v != 0:
                return False
        else:  # strings (e.g. molecular formula) are never "zero"
            return False
    return True


def _count_index_row(
    mqd: MolecularIQD,
    rng: random.Random,
    task: str,
    mol: dict,
    constructs: list[str],
    cmap: dict[str, list[str]],
    num_items: int,
) -> Optional[dict]:
    smiles = mol["smiles"]
    chosen_constructs = (
        [rng.choice(constructs)]
        if num_items == 1
        else rng.sample(constructs, min(num_items, len(constructs)))
    )
    props = [rng.choice(cmap[c]) for c in chosen_constructs]

    if task == TASK_COUNT:
        question, target, _meta = mqd.generate_count_question(smiles, props)
    else:
        question, target, _meta = mqd.generate_index_question(smiles, props)

    if num_items == 1:
        features = f"single_{task}_{chosen_constructs[0]}"
    else:
        features = f"multi_{task}_nbr_{len(props)}"

    return {
        "task_type": _TASK_TYPE_FIELD[task],
        "features": features,
        "question": question,
        "target": json.dumps(target),
        "constraints": None,
        "original_smiles": smiles,
        "complexity_bin": mol["complexity_bin"],
        "multi_task_load": len(props),
        "metadata": json.dumps(
            {"smiles": smiles, "is_randomized": False, "is_kekulized": False,
             "properties": props}
        ),
        "_target_dict": target,       # internal, stripped before writing
        "_dedup_key": (smiles, tuple(sorted(props))),
    }


# ---------------------------------------------------------------------------
# Constraint question generation
# ---------------------------------------------------------------------------

_NUMERIC_OPERATORS = ("=", ">=", "<=", ">", "<", "range")


def _anchor_property_value(mqd: MolecularIQD, smiles: str, prop: str) -> Any:
    """Property value of the anchor molecule, with constraint semantics.

    Functional-group constraints are verified against the number of group
    *instances* (nbrInstances) by the official verifier, whereas count
    questions use the number of atoms — so anchor on nbrInstances here.
    """
    if prop.startswith("functional_group_") and prop.endswith("_count"):
        group = prop[len("functional_group_"):-len("_count")]
        data = mqd.solver.functional_group_solver.get_counts_and_indices(smiles)
        return data.get(f"functional_group_{group}_nbrInstances", 0)
    return mqd.compute_property(smiles, prop)


def is_vacuous_constraint(constraint: dict) -> bool:
    """True if *every* valid molecule satisfies the constraint.

    MolecularIQ count properties are non-negative, so ``>= 0`` asks nothing of
    the answer: any parseable molecule scores 1.0. Such questions teach a
    generative policy that emitting anything is correct, which is precisely the
    reward-hacking surface that let a constant string score 0.53 on our
    generated constraint set.
    """
    op = constraint.get("operator")
    value = constraint.get("value")
    if op == ">=" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return value <= 0
    if op == "range":
        lo = constraint.get("min_value")
        if isinstance(lo, (int, float)) and lo <= 0:
            # A range anchored at 0 is only vacuous if it also has no upper
            # bound; the generator always sets a finite, small max_value, so
            # this is here to stay correct if that ever changes.
            hi = constraint.get("max_value")
            return hi is None
    return False


# How many times to resample the operator before giving up on a property.
# Vacuous constraints arise only from `>=` against a small anchor value, so a
# handful of draws is plenty.
_VACUOUS_RETRIES = 8


def _sample_numeric_constraint(rng: random.Random, prop: str, v) -> Optional[dict]:
    """One draw of an operator + loosened bound around the anchor value ``v``."""
    op = rng.choice(_NUMERIC_OPERATORS)
    delta = max(1, int(abs(v) * 0.3))

    if op == "=":
        return {"property": prop, "operator": "=", "value": v}
    if op == ">=":
        return {"property": prop, "operator": ">=", "value": max(0, v - rng.randint(0, delta))}
    if op == "<=":
        return {"property": prop, "operator": "<=", "value": v + rng.randint(0, delta)}
    if op == ">":
        if v < 1:
            return {"property": prop, "operator": "=", "value": v}
        return {"property": prop, "operator": ">", "value": max(0, v - rng.randint(1, delta))}
    if op == "<":
        return {"property": prop, "operator": "<", "value": v + rng.randint(1, delta)}
    # range
    lo = max(0, v - rng.randint(0, delta))
    hi = v + rng.randint(0, delta)
    if lo == hi:
        return {"property": prop, "operator": "=", "value": v}
    return {
        "property": prop, "operator": "range", "value": None,
        "min_value": lo, "max_value": hi,
    }


def _build_constraint(
    rng: random.Random, prop: str, value: Any
) -> Optional[dict]:
    """Build a constraint dict the anchor molecule satisfies.

    Rejects vacuous constraints (see :func:`is_vacuous_constraint`) by
    resampling the operator, and gives up on the property if every draw is
    vacuous, so the row is regenerated rather than silently trivial.
    """
    if isinstance(value, bool):
        return {"property": prop, "operator": "=", "value": int(value)}

    if isinstance(value, str):
        if not value:
            return None
        return {"property": prop, "operator": "=", "value": value}

    if not isinstance(value, (int, float)):
        return None

    v = int(value) if float(value).is_integer() else float(value)
    for _ in range(_VACUOUS_RETRIES):
        constraint = _sample_numeric_constraint(rng, prop, v)
        if constraint is not None and not is_vacuous_constraint(constraint):
            return constraint
    return None


def _constraint_row(
    mqd: MolecularIQD,
    rng: random.Random,
    mol: dict,
    constructs: list[str],
    cmap: dict[str, list[str]],
    num_items: int,
) -> Optional[dict]:
    smiles = mol["smiles"]
    chosen = rng.sample(constructs, min(num_items, len(constructs)))
    constraint_dicts: list[dict] = []
    zero_anchored = False

    for construct in chosen:
        prop = rng.choice(cmap[construct])
        try:
            value = _anchor_property_value(mqd, smiles, prop)
        except Exception:
            return None
        if value is None or isinstance(value, (list, dict)):
            return None
        # An anchor value of 0 makes every operator branch satisfiable by any
        # molecule that also lacks the property -- which is most molecules, for
        # the exotic constructs (bridgehead atoms, E/Z bonds, stereocenters)
        # where 0 is the common value. Flagged here so generate_split can cap
        # the share of such questions, exactly as it does for count/index.
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
            zero_anchored = True
        constraint = _build_constraint(rng, prop, value)
        if constraint is None:
            return None
        constraint_dicts.append(constraint)

    # The anchor molecule must pass the official verifier — this guarantees
    # the question is satisfiable AND that every property is supported.
    try:
        if mqd.validate_constraint_answer(smiles, constraint_dicts) != 1.0:
            return None
    except Exception:
        return None

    question, _meta = mqd.generate_constraint_question(constraint_dicts)

    if num_items == 1:
        features = f"single_constraint_generation_{chosen[0]}"
    else:
        features = f"multi_constraint_generation_{len(constraint_dicts)}_constraints"

    constraints_json = json.dumps(constraint_dicts)
    return {
        "task_type": _TASK_TYPE_FIELD[TASK_CONSTRAINT],
        "features": features,
        "question": question,
        "target": None,
        "constraints": constraints_json,
        "original_smiles": "",
        "complexity_bin": mol["complexity_bin"],
        "multi_task_load": len(constraint_dicts),
        "metadata": json.dumps(
            {"is_randomized": None, "is_kekulized": None, "anchor_smiles": smiles}
        ),
        "_target_dict": None,
        "_dedup_key": constraints_json,
        "_zero_anchored": zero_anchored,
    }


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def generate_split(
    pool_rows: list[dict],
    n_questions: int,
    cfg: GenerationConfig,
    uid_prefix: str,
    seed_offset: int = 0,
) -> list[dict]:
    """Generate ``n_questions`` questions of one task type from a molecule set."""
    if cfg.task not in TASKS:
        raise ValueError(f"Unknown task {cfg.task!r}; expected one of {TASKS}")

    seed = cfg.seed + seed_offset
    rng = random.Random(seed)
    mqd = MolecularIQD(seed=seed, enable_random_phrasing=True)
    cmap = build_construct_map(cfg.task, mqd)

    constructs = cfg.constructs or sorted(cmap.keys())
    unknown = [c for c in constructs if c not in cmap]
    if unknown:
        raise ValueError(
            f"Unknown construct(s) for task {cfg.task!r}: {unknown}. "
            f"Available: {sorted(cmap.keys())}"
        )

    sampler = _BinSampler(rng=rng, by_bin=_group_by_bin(pool_rows))

    rows: list[dict] = []
    seen: set = set()
    zero_answers = 0
    attempts = 0
    max_attempts = max(1000, n_questions * cfg.max_attempts_factor)

    while len(rows) < n_questions and attempts < max_attempts:
        attempts += 1
        mol = sampler.next_molecule()
        try:
            if cfg.task == TASK_CONSTRAINT:
                row = _constraint_row(mqd, rng, mol, constructs, cmap, cfg.num_items)
            else:
                row = _count_index_row(
                    mqd, rng, cfg.task, mol, constructs, cmap, cfg.num_items
                )
        except Exception:
            logger.debug("Question generation failed for %s", mol["smiles"], exc_info=True)
            continue
        if row is None or row["_dedup_key"] in seen:
            continue

        # Cap trivially-satisfiable questions. For count/index that means a
        # zero/empty answer; for constraint generation it means a constraint
        # anchored on a property the molecule has none of, which any molecule
        # lacking that property also satisfies. Both let a policy score without
        # reasoning, so both are capped by the same fraction.
        if cfg.task in (TASK_COUNT, TASK_INDEX):
            trivial = _is_zero_answer(row["_target_dict"])
        else:
            trivial = bool(row.get("_zero_anchored"))
        if trivial:
            if zero_answers >= cfg.max_zero_fraction * n_questions:
                continue
            zero_answers += 1

        seen.add(row["_dedup_key"])
        row.pop("_target_dict")
        row.pop("_dedup_key")
        row.pop("_zero_anchored", None)
        row["uid"] = f"{uid_prefix}_{len(rows):08d}"
        rows.append(row)

        # MolecularIQD caches every computed property; for large runs this
        # grows unbounded, so clear it periodically.
        if len(rows) % 2000 == 0:
            mqd.clear_cache()

    if len(rows) < n_questions:
        logger.warning(
            "Generated only %d/%d questions after %d attempts "
            "(constraints too strict / pool too small?)",
            len(rows), n_questions, attempts,
        )

    # uid first, official column order
    ordered_cols = [
        "uid", "task_type", "features", "question", "target", "constraints",
        "original_smiles", "complexity_bin", "multi_task_load", "metadata",
    ]
    return [{c: r[c] for c in ordered_cols} for r in rows]


def load_pool(cache_dir: Optional[str] = None) -> list[dict]:
    """Load the MolecularIQ training molecule pool from Hugging Face."""
    from datasets import load_dataset

    ds = load_dataset("ml-jku/moleculariq-trainPool", split="train", cache_dir=cache_dir)
    cols = ["smiles", "complexity_bin"]
    ds = ds.select_columns(cols)
    return [{"smiles": s, "complexity_bin": b}
            for s, b in zip(ds["smiles"], ds["complexity_bin"])]


def write_jsonl(rows: list[dict], path) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
