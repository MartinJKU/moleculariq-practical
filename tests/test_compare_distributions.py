"""Unit tests for scripts/compare_distributions.py.

The parsing helpers are the risky part: a silently wrong heavy-atom count or a
SMILES regex that matches English words would produce a confident, wrong
distribution comparison. Expected values here are counted by hand.

Run with:  pytest -q
"""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_distributions.py"
_spec = importlib.util.spec_from_file_location("compare_distributions", _PATH)
cd = importlib.util.module_from_spec(_spec)
sys.modules["compare_distributions"] = cd
_spec.loader.exec_module(cd)


# --------------------------------------------------------------------------
# heavy_atom_count
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("CCO", 3),                 # ethanol
    ("CC(C)O", 4),              # isopropanol
    ("CC(=O)N", 4),             # acetamide
    ("c1ccccc1", 6),            # benzene
    ("CC(=O)C", 4),             # acetone
    ("ClCCBr", 4),              # two-letter symbols must not double-count
    ("C1CCCCC1", 6),            # cyclohexane; ring-closure digits are not atoms
    ("[nH]1cccc1", 5),          # pyrrole: bracketed aromatic N + 4 carbons
    ("C[C@@H](N)C(=O)O", 6),    # alanine: stereo brackets are one atom each
])
def test_heavy_atom_count(smiles, expected):
    assert cd.heavy_atom_count(smiles) == expected


def test_explicit_hydrogen_is_skipped_but_isotopes_count():
    # The official prompt: skip [H], include [2H]/[3H].
    assert cd.heavy_atom_count("C[H]") == 1
    assert cd.heavy_atom_count("C[2H]") == 2


def test_heavy_atom_count_handles_empty_and_none():
    assert cd.heavy_atom_count(None) is None
    assert cd.heavy_atom_count("") is None


def test_heavy_atom_count_survives_malformed_smiles():
    assert cd.heavy_atom_count("CC[") == 2      # unterminated bracket


# --------------------------------------------------------------------------
# SMILES detection (the false-positive risk)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word", [
    "molecule", "structure", "aromatic", "compound", "contains",
    "following", "hydrogen", "indices", "property",
])
def test_english_words_are_not_smiles(word):
    assert not cd._looks_like_smiles(word)


@pytest.mark.parametrize("smiles", [
    "CC(=O)Nc1ccccc1", "C1CCCCC1", "CCOC(=O)C", "c1ccc2ccccc2c1",
])
def test_real_smiles_are_detected(smiles):
    assert cd._looks_like_smiles(smiles)


def test_extract_smiles_prefers_explicit_field():
    row = {"original_smiles": "CCO", "question": "How many rings in CC(=O)N?"}
    assert cd.extract_smiles(row) == "CCO"


def test_extract_smiles_reads_metadata_json():
    row = {"metadata": '{"smiles": "CCN", "is_randomized": false}'}
    assert cd.extract_smiles(row) == "CCN"


def test_extract_smiles_falls_back_to_question_text():
    row = {"question": "How many aromatic rings does CC(=O)Nc1ccccc1 contain?"}
    assert cd.extract_smiles(row) == "CC(=O)Nc1ccccc1"


def test_extract_smiles_returns_none_when_absent():
    assert cd.extract_smiles({"question": "How many rings are there?"}) is None


# --------------------------------------------------------------------------
# answer_magnitude
# --------------------------------------------------------------------------

def test_answer_magnitude_count():
    assert cd.answer_magnitude({"target": '{"ring_count": 3}'}, "count") == 3


def test_answer_magnitude_count_accepts_parsed_dict():
    assert cd.answer_magnitude({"target": {"ring_count": 7}}, "count") == 7


def test_answer_magnitude_index_uses_list_length():
    row = {"target": '{"carbon_indices": [0, 2, 5, 9]}'}
    assert cd.answer_magnitude(row, "index") == 4


def test_answer_magnitude_zero_is_kept_not_treated_as_missing():
    """A zero answer is real data; conflating it with 'missing' would bias the
    magnitude histogram exactly where the zero-answer cap matters."""
    assert cd.answer_magnitude({"target": '{"ring_count": 0}'}, "count") == 0


def test_answer_magnitude_handles_missing_and_malformed():
    assert cd.answer_magnitude({}, "count") is None
    assert cd.answer_magnitude({"target": None}, "count") is None
    assert cd.answer_magnitude({"target": "not json"}, "count") is None
    assert cd.answer_magnitude({"target": "{}"}, "count") is None


def test_answer_magnitude_ignores_booleans():
    assert cd.answer_magnitude({"target": '{"has_ring": true}'}, "count") is None


# --------------------------------------------------------------------------
# Distribution distance
# --------------------------------------------------------------------------

def test_tvd_identical_is_zero():
    a = Counter({"x": 10, "y": 10})
    assert cd.total_variation(a, Counter({"x": 20, "y": 20})) == pytest.approx(0.0)


def test_tvd_disjoint_is_one():
    assert cd.total_variation(Counter({"x": 5}), Counter({"y": 5})) == pytest.approx(1.0)


def test_tvd_is_scale_invariant():
    a, b = Counter({"x": 3, "y": 1}), Counter({"x": 300, "y": 100})
    assert cd.total_variation(a, b) == pytest.approx(0.0)


def test_tvd_partial_overlap():
    # 50/50 vs 100/0 -> TVD 0.5
    a, b = Counter({"x": 5, "y": 5}), Counter({"x": 10})
    assert cd.total_variation(a, b) == pytest.approx(0.5)


def test_tvd_empty_is_nan():
    v = cd.total_variation(Counter(), Counter({"x": 1}))
    assert v != v


def test_histogram_tvd_bins_values():
    bins = [(0, 0), (1, 3), (4, 10**6)]
    # all zeros vs all large -> disjoint
    assert cd.histogram_tvd([0] * 10, [50] * 10, bins) == pytest.approx(1.0)
    assert cd.histogram_tvd([2] * 10, [3] * 10, bins) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------

def test_profile_collects_all_three_axes():
    rows = [
        {"features": "single_count_ring", "target": '{"ring_count": 2}',
         "original_smiles": "C1CCCCC1"},
        {"features": "single_count_carbon", "target": '{"carbon_count": 6}',
         "original_smiles": "CCCCCC"},
    ]
    p = cd.profile(rows, "count")
    assert p["n"] == 2
    assert p["features"]["single_count_ring"] == 1
    assert sorted(p["magnitudes"]) == [2, 6]
    assert sorted(p["sizes"]) == [6, 6]


def test_profile_labels_missing_features_as_unknown():
    p = cd.profile([{"features": None, "target": '{"c": 1}'}], "count")
    assert p["features"]["unknown"] == 1


def test_numeric_summary():
    s = cd.numeric_summary(list(range(1, 101)))
    assert s["n"] == 100
    assert s["min"] == 1 and s["max"] == 100
    assert s["mean"] == pytest.approx(50.5)


def test_numeric_summary_empty():
    assert cd.numeric_summary([]) == {"n": 0}
