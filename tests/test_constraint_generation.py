"""Unit tests for the constraint-generation fixes in moleculariq_grpo.data.

Background: the original generator produced constraint questions that a
constant answer could satisfy. A fixed string scored 0.530 on our generated
constraint set against 0.540 for the untrained baseline, and a Monte-Carlo
replication showed that when the anchor property value is 0, *every* operator
branch yields a constraint any zero-valued molecule satisfies, with ~16.5%
collapsing to a vacuous ``>= 0``.

These tests pin the two fixes: vacuous constraints are never emitted, and
zero-anchored constraints are flagged so the generator can cap their share.

``moleculariq_core`` is not importable in every environment (it needs RDKit),
so the pure functions are loaded directly from the module source rather than
by importing the package.

Run with:  pytest -q
"""
import ast
import random
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "moleculariq_grpo" / "data.py"


def _load_pure_functions():
    """Exec only the dependency-free helpers from data.py.

    data.py imports moleculariq_core at module scope, which pulls in RDKit; the
    functions under test touch neither, so the relevant defs are extracted from
    the AST and executed in isolation.
    """
    tree = ast.parse(_SRC.read_text())
    wanted = {"is_vacuous_constraint", "_sample_numeric_constraint",
              "_build_constraint", "_is_zero_answer"}
    # data.py uses `from __future__ import annotations`, which is not carried
    # over by exec'ing individual defs, so the annotations are evaluated eagerly
    # and need the real typing objects.
    from typing import Any, Optional
    ns: dict = {"random": random, "Optional": Optional, "Any": Any}
    body = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assign = [n for n in tree.body
              if isinstance(n, ast.Assign)
              and any(getattr(t, "id", "") in {"_NUMERIC_OPERATORS", "_VACUOUS_RETRIES"}
                      for t in n.targets)]
    module = ast.Module(body=assign + body, type_ignores=[])
    exec(compile(module, str(_SRC), "exec"), ns)                    # noqa: S102
    missing = wanted - set(ns)
    if missing:
        raise AssertionError(f"could not load {missing} from {_SRC}")
    return ns


D = _load_pure_functions()
is_vacuous_constraint = D["is_vacuous_constraint"]
_build_constraint = D["_build_constraint"]


# --------------------------------------------------------------------------
# is_vacuous_constraint
# --------------------------------------------------------------------------

def test_ge_zero_is_vacuous():
    """`>= 0` is satisfied by every molecule: counts are non-negative."""
    assert is_vacuous_constraint({"property": "ring_count", "operator": ">=", "value": 0})


def test_ge_negative_is_vacuous():
    assert is_vacuous_constraint({"property": "p", "operator": ">=", "value": -1})


def test_ge_positive_is_not_vacuous():
    assert not is_vacuous_constraint({"property": "p", "operator": ">=", "value": 1})


@pytest.mark.parametrize("constraint", [
    {"property": "p", "operator": "=", "value": 0},      # "exactly zero" is a real ask
    {"property": "p", "operator": "<=", "value": 0},     # so is "at most zero"
    {"property": "p", "operator": "<", "value": 1},
    {"property": "p", "operator": ">", "value": 0},
    {"property": "p", "operator": "range", "min_value": 0, "max_value": 2},
    {"property": "p", "operator": "=", "value": "C6H6"},  # formula match
])
def test_non_vacuous_constraints(constraint):
    assert not is_vacuous_constraint(constraint)


def test_open_ended_range_is_vacuous():
    assert is_vacuous_constraint(
        {"property": "p", "operator": "range", "min_value": 0, "max_value": None})


def test_boolean_value_is_not_treated_as_numeric_zero():
    """`>= False` must not be read as `>= 0` and silently dropped."""
    assert not is_vacuous_constraint({"property": "p", "operator": ">=", "value": False})


# --------------------------------------------------------------------------
# _build_constraint never emits a vacuous constraint
# --------------------------------------------------------------------------

def test_build_constraint_never_vacuous_for_zero_anchor():
    """The motivating case: anchor value 0 previously produced `>= 0` ~16% of
    the time. Over many draws none may survive now."""
    rng = random.Random(0)
    built = [_build_constraint(rng, "bridgehead_count", 0) for _ in range(3000)]
    assert any(c is not None for c in built), "generator produced nothing at all"
    assert not any(c is not None and is_vacuous_constraint(c) for c in built)


@pytest.mark.parametrize("anchor", [0, 1, 2, 5, 17])
def test_build_constraint_never_vacuous_for_any_anchor(anchor):
    rng = random.Random(anchor)
    for _ in range(500):
        c = _build_constraint(rng, "ring_count", anchor)
        assert c is None or not is_vacuous_constraint(c)


def test_build_constraint_still_produces_variety():
    """Rejecting vacuous draws must not collapse the operator distribution."""
    rng = random.Random(1)
    ops = {c["operator"] for _ in range(2000)
           if (c := _build_constraint(rng, "ring_count", 4)) is not None}
    assert len(ops) >= 4, f"operator variety collapsed: {ops}"


def test_anchor_still_satisfies_the_constraint():
    """Whatever is emitted, the anchor molecule's value must satisfy it --
    otherwise the question is unsatisfiable by construction."""
    rng = random.Random(7)
    for anchor in (0, 1, 3, 9):
        for _ in range(300):
            c = _build_constraint(rng, "p", anchor)
            if c is None:
                continue
            op, val = c["operator"], c.get("value")
            if op == "=":
                assert anchor == val
            elif op == ">=":
                assert anchor >= val
            elif op == "<=":
                assert anchor <= val
            elif op == ">":
                assert anchor > val
            elif op == "<":
                assert anchor < val
            else:
                assert c["min_value"] <= anchor <= c["max_value"]


def test_string_and_bool_values_unchanged():
    rng = random.Random(0)
    assert _build_constraint(rng, "formula", "C6H6") == {
        "property": "formula", "operator": "=", "value": "C6H6"}
    assert _build_constraint(rng, "flag", True) == {
        "property": "flag", "operator": "=", "value": 1}
    assert _build_constraint(rng, "formula", "") is None
    assert _build_constraint(rng, "p", None) is None
