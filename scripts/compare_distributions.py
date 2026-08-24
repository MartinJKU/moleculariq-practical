#!/usr/bin/env python
"""Compare a generated training set against the official benchmark split.

Motivation: our count-trained model improves on our own held-out questions but
degrades on the official ``single_count`` split. A policy that gets better
in-distribution while getting worse on the benchmark has fitted the generator
rather than the task -- so the question is *where* the two distributions differ.

This script answers that on CPU, without touching a GPU or a model. It compares
three axes that plausibly drive counting difficulty:

* **construct mix** -- which ``features`` the questions ask about. Difficulty
  varies enormously by construct (counting rings is far easier than counting
  carbon atoms), so a different mix alone can flip a result.
* **answer magnitude** -- the integer answer for counts, or the list length for
  indices. Counting to 3 and counting to 27 are different tasks.
* **molecule size** -- heavy-atom count of the referenced molecule, with
  question length as a fallback when no SMILES is recoverable.

For each axis it reports a summary and the total variation distance (TVD)
between the two distributions, and writes a comparison figure.

Examples
--------
python scripts/compare_distributions.py --task count \
    --generated data/count/train.jsonl \
    --official-split single_count --out results/dist/count

python scripts/compare_distributions.py --task index \
    --generated data/index/train.jsonl \
    --official-split single_index --out results/dist/index
"""
import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_distributions")

# Palette (shared with compare_evals.py): generated vs official as two series.
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, AXIS = "#e1e0d9", "#c3c2b7"
C_GENERATED, C_OFFICIAL = "#2a78d6", "#eb6834"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["count", "index"], default="count",
                   help="Answer-magnitude semantics: integer value vs list length")
    p.add_argument("--generated", type=Path, required=True,
                   help="Our generated .jsonl (e.g. data/count/train.jsonl)")
    p.add_argument("--official-dataset", default="ml-jku/moleculariq-v0.0")
    p.add_argument("--official-split", default="single_count")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap rows read from each source (for a quick look)")
    p.add_argument("--out", type=Path, default=Path("results/dist/count"),
                   help="Output prefix: writes <out>.png/.pdf and <out>.json")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_generated(path: Path, limit=None) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_official(dataset: str, split: str, limit=None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(r) for r in ds]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Heavy atoms: element symbols outside brackets, plus every bracketed atom.
# Two-letter symbols are listed first so they win the alternation over B/C/S.
_ORGANIC = re.compile(r"Br|Cl|Si|Se|[BCNOPSFIbcnops]")

# Characters that may appear in a SMILES string. Used to reject English words
# when falling back to scraping a SMILES out of the question text -- note that
# a, e, t, u, d, g, h, m, v, w, x and y are absent, which rules out essentially
# all prose.
_SMILES_CHARS = set("BCNOPSFIbcnopslr0123456789()[]=#$%/\\+-@.*")
_TOKEN_RE = re.compile(r"\S{6,}")


def _looks_like_smiles(token: str) -> bool:
    """Best-effort test that a token is a SMILES string rather than a word."""
    if not all(ch in _SMILES_CHARS for ch in token):
        return False
    # Require real SMILES structure, or enough length that a false positive
    # from the character set alone is implausible.
    return any(ch in "()[]=#123456789" for ch in token) or len(token) >= 8


def extract_smiles(row: dict) -> str | None:
    """Best-effort SMILES recovery, tolerant of schema differences."""
    for key in ("original_smiles", "smiles"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    meta = row.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, ValueError):
            meta = None
    if isinstance(meta, dict):
        for key in ("smiles", "anchor_smiles"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    # Fall back to the longest SMILES-shaped token in the question text.
    q = row.get("question") or ""
    cands = [c.strip(".,;:'\"") for c in _TOKEN_RE.findall(q)]
    cands = [c for c in cands if len(c) >= 6 and _looks_like_smiles(c)]
    return max(cands, key=len) if cands else None


def heavy_atom_count(smiles: str | None) -> int | None:
    """Approximate heavy-atom count from a SMILES string.

    Deliberately regex-based rather than RDKit: this script must run on a login
    node where the chemistry stack may be unavailable, and an approximate size
    is enough to compare two distributions.
    """
    if not smiles:
        return None
    n = 0
    i = 0
    while i < len(smiles):
        if smiles[i] == "[":                       # bracketed atom: one heavy atom
            j = smiles.find("]", i)
            if j == -1:
                break
            # Per the official prompt: skip a plain [H], but count the
            # isotopes [2H]/[3H] and every other bracketed atom ([nH], [C@@H]).
            if not re.fullmatch(r"H\d*", smiles[i + 1:j]):
                n += 1
            i = j + 1
            continue
        m = _ORGANIC.match(smiles, i)
        if m:
            n += 1
            i = m.end()
        else:
            i += 1
    return n or None


def answer_magnitude(row: dict, task: str) -> int | None:
    """Integer answer for counts; list length for indices."""
    target = row.get("target")
    if isinstance(target, str):
        try:
            target = json.loads(target)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(target, dict) or not target:
        return None
    values = list(target.values())
    if task == "index":
        lists = [v for v in values if isinstance(v, list)]
        return len(lists[0]) if lists else None
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return int(nums[0]) if nums else None


def profile(rows: list[dict], task: str) -> dict:
    feats, mags, sizes = Counter(), [], []
    for r in rows:
        feats[r.get("features") or "unknown"] += 1
        m = answer_magnitude(r, task)
        if m is not None:
            mags.append(m)
        h = heavy_atom_count(extract_smiles(r))
        if h is not None:
            sizes.append(h)
    return {"n": len(rows), "features": feats, "magnitudes": mags, "sizes": sizes}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def total_variation(a: Counter, b: Counter) -> float:
    """TVD in [0,1]: 0 = identical mixes, 1 = disjoint."""
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return float("nan")
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys)


def numeric_summary(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    def q(p):
        return s[min(len(s) - 1, int(p * len(s)))]
    return {"n": len(s), "mean": sum(s) / len(s), "median": q(0.5),
            "p10": q(0.10), "p90": q(0.90), "min": s[0], "max": s[-1]}


def histogram_tvd(a: list[int], b: list[int], bins: list[tuple]) -> float:
    def binned(vals):
        c = Counter()
        for v in vals:
            for lo, hi in bins:
                if lo <= v <= hi:
                    c[(lo, hi)] += 1
                    break
        return c
    return total_variation(binned(a), binned(b))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.titlecolor": INK, "font.size": 9,
        "grid.color": GRIDLINE, "grid.linewidth": 0.8,
    })
    return plt


def _style(ax):
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)


def plot_comparison(plt, gen, off, task, tvds, out: Path, formats):
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)

    # --- construct mix ----------------------------------------------------
    ax = axes[0]
    keys = [k for k, _ in (gen["features"] + off["features"]).most_common(12)]
    ng, no = sum(gen["features"].values()), sum(off["features"].values())
    y = np.arange(len(keys))
    short = [k.replace("single_count_", "").replace("single_index_", "")[:22]
             for k in keys]
    ax.barh(y - 0.2, [gen["features"][k] / max(1, ng) for k in keys], height=0.36,
            color=C_GENERATED, label="generated (train)", zorder=3)
    ax.barh(y + 0.2, [off["features"][k] / max(1, no) for k in keys], height=0.36,
            color=C_OFFICIAL, label="official (benchmark)", zorder=3)
    ax.set_yticks(y, labels=short)
    ax.invert_yaxis()
    ax.set_xlabel("share of questions")
    ax.set_title(f"Construct mix  (TVD {tvds['features']:.2f})")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    ax.grid(axis="x", zorder=0)

    # --- answer magnitude -------------------------------------------------
    ax = axes[1]
    label = "answer value" if task == "count" else "number of indices"
    allv = gen["magnitudes"] + off["magnitudes"]
    hi = int(np.percentile(allv, 99)) if allv else 10
    bins = np.arange(0, max(2, hi + 2)) - 0.5
    for vals, c, lab in ((gen["magnitudes"], C_GENERATED, "generated"),
                         (off["magnitudes"], C_OFFICIAL, "official")):
        if vals:
            ax.hist(vals, bins=bins, density=True, color=c, alpha=0.62,
                    label=lab, zorder=3)
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.set_title(f"Answer magnitude  (TVD {tvds['magnitude']:.2f})")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    # --- molecule size ----------------------------------------------------
    ax = axes[2]
    alls = gen["sizes"] + off["sizes"]
    if alls:
        hi = int(np.percentile(alls, 99))
        bins = np.linspace(0, max(10, hi), 26)
        for vals, c, lab in ((gen["sizes"], C_GENERATED, "generated"),
                             (off["sizes"], C_OFFICIAL, "official")):
            if vals:
                ax.hist(vals, bins=bins, density=True, color=c, alpha=0.62,
                        label=lab, zorder=3)
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no SMILES recoverable", ha="center", va="center",
                color=MUTED, transform=ax.transAxes)
    ax.set_xlabel("heavy atoms (approx.)")
    ax.set_ylabel("density")
    ax.set_title(f"Molecule size  (TVD {tvds['size']:.2f})")
    _style(ax)

    fig.suptitle("Generated training questions vs official benchmark split",
                 color=INK, fontsize=12)
    fig.text(0.005, -0.02, "TVD = total variation distance (0 = identical, "
                           "1 = disjoint). Large values mean the model was "
                           "trained on a different task than it is tested on.",
             color=MUTED, fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
        logger.info("wrote %s", out.with_suffix(f".{ext}"))
    plt.close(fig)


def main():
    args = parse_args()
    logger.info("Loading generated: %s", args.generated)
    gen_rows = load_generated(args.generated, args.limit)
    logger.info("Loading official: %s split=%s", args.official_dataset,
                args.official_split)
    off_rows = load_official(args.official_dataset, args.official_split, args.limit)

    gen, off = profile(gen_rows, args.task), profile(off_rows, args.task)
    logger.info("generated: %d rows | official: %d rows", gen["n"], off["n"])

    mag_bins = [(0, 0), (1, 1), (2, 3), (4, 6), (7, 10), (11, 20), (21, 10**6)]
    size_bins = [(0, 10), (11, 20), (21, 30), (31, 45), (46, 70), (71, 10**6)]
    tvds = {
        "features": total_variation(gen["features"], off["features"]),
        "magnitude": histogram_tvd(gen["magnitudes"], off["magnitudes"], mag_bins),
        "size": histogram_tvd(gen["sizes"], off["sizes"], size_bins),
    }

    report = {
        "task": args.task,
        "generated": str(args.generated),
        "official": f"{args.official_dataset}:{args.official_split}",
        "n_generated": gen["n"], "n_official": off["n"],
        "tvd": tvds,
        "magnitude_generated": numeric_summary(gen["magnitudes"]),
        "magnitude_official": numeric_summary(off["magnitudes"]),
        "size_generated": numeric_summary(gen["sizes"]),
        "size_official": numeric_summary(off["sizes"]),
        "features_generated": dict(gen["features"].most_common()),
        "features_official": dict(off["features"].most_common()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", args.out.with_suffix(".json"))

    plot_comparison(setup_plt(), gen, off, args.task, tvds, args.out, args.formats)

    logger.info("--- distribution gap ---")
    for name, v in tvds.items():
        flag = "  <-- LARGE" if v == v and v > 0.30 else ""
        logger.info("  TVD %-10s %.3f%s", name, v, flag)
    logger.info("  answer magnitude  generated %s", report["magnitude_generated"])
    logger.info("  answer magnitude  official  %s", report["magnitude_official"])
    logger.info("  molecule size     generated %s", report["size_generated"])
    logger.info("  molecule size     official  %s", report["size_official"])
    if any(v == v and v > 0.30 for v in tvds.values()):
        logger.warning(
            "A TVD above ~0.3 on any axis means training and evaluation are "
            "meaningfully different tasks. Rebalance the generator on that axis "
            "before drawing conclusions about what training did or did not learn.")


if __name__ == "__main__":
    main()
