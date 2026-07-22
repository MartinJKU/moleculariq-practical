#!/usr/bin/env python
"""Run the model x dataset eval matrix and plot baseline-vs-trained comparisons.

Evaluates every model in the config on every eval dataset in the config (the
full cross product) by calling scripts/evaluate.py per cell, then renders
heatmaps / bar charts / breakdown plots plus a CSV+Markdown summary. Covers
both questions of the single-task practical work in one grid:

  * specialization: task-trained model vs baseline on its own task
  * generalization: task-trained model on the *other* tasks
    (e.g. the count model evaluated on index questions)

Existing result files are skipped, so the matrix fills in incrementally as
trainings finish — re-run the same command any time.

Examples
--------
# Everything (evals + plots), driven by the matrix config:
python scripts/compare_evals.py all --config configs/eval_matrix.yaml

# Show what would run, without running it:
python scripts/compare_evals.py run --config configs/eval_matrix.yaml --dry-run

# Only the missing cells for one model:
python scripts/compare_evals.py run --config configs/eval_matrix.yaml --models count

# Re-plot from existing results (no GPU needed, fine on a login node):
python scripts/compare_evals.py plot --config configs/eval_matrix.yaml

# Just one group (each group renders into its own plots subfolder):
python scripts/compare_evals.py all --config configs/eval_matrix.yaml --groups heldout

# On Leonardo:
sbatch slurm/eval_matrix.slurm
"""
import argparse
import csv
import json
import logging
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_evals")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# eval-config key -> evaluate.py flag
EVAL_FLAGS = [
    ("split", "--split"),
    ("n", "--n"),
    ("temperature", "--temperature"),
    ("top_p", "--top-p"),
    ("limit", "--limit"),
    ("max_tokens", "--max-tokens"),
    ("max_model_len", "--max-model-len"),
    ("gpu_mem_util", "--gpu-mem-util"),
    ("tensor_parallel", "--tensor-parallel"),
    ("dtype", "--dtype"),
    ("seed", "--seed"),
]

KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for section in ("models", "evals"):
        if not cfg.get(section):
            raise SystemExit(f"config {path}: missing/empty '{section}' section")
        for key in cfg[section]:
            if not KEY_RE.match(key):
                raise SystemExit(f"config {path}: key '{key}' must match {KEY_RE.pattern} "
                                 "(it is used in result filenames)")
    defaults = cfg.get("eval_defaults") or {}
    cfg["evals"] = {k: {**defaults, **(v or {})} for k, v in cfg["evals"].items()}
    for key, m in cfg["models"].items():
        m.setdefault("label", key)
    baseline = cfg.get("baseline")
    if baseline is not None and baseline not in cfg["models"]:
        raise SystemExit(f"config {path}: baseline '{baseline}' is not a key of 'models'")
    cfg["results_dir"] = Path(cfg.get("results_dir", "results/matrix"))
    cfg["plots_dir"] = Path(cfg.get("plots_dir", cfg["results_dir"] / "plots"))
    cfg.setdefault("metric", "avg_accuracy")

    # `groups` decide what is compared/plotted together, each into its own
    # subfolder. With no `groups`, fall back to a single group over everything.
    groups = cfg.get("groups") or {
        "all": {"models": list(cfg["models"]), "evals": list(cfg["evals"])}}
    norm = {}
    for gkey, g in groups.items():
        if not KEY_RE.match(gkey):
            raise SystemExit(f"config {path}: group '{gkey}' must match {KEY_RE.pattern}")
        g = dict(g or {})
        g["models"] = g.get("models") or list(cfg["models"])
        g["evals"] = g.get("evals") or list(cfg["evals"])
        for kind, keys, registry in (("model", g["models"], cfg["models"]),
                                     ("eval", g["evals"], cfg["evals"])):
            unknown = [k for k in keys if k not in registry]
            if unknown:
                raise SystemExit(f"config {path}: group '{gkey}' references unknown "
                                 f"{kind}(s) {unknown}; known: {sorted(registry)}")
        g.setdefault("metric", cfg["metric"])
        g.setdefault("title", gkey)
        subdir = g.setdefault("subdir", gkey)
        if not KEY_RE.match(str(subdir)):
            raise SystemExit(f"config {path}: group '{gkey}' subdir '{subdir}' "
                             f"must match {KEY_RE.pattern}")
        norm[gkey] = g
    cfg["groups"] = norm
    return cfg


def group_cells(cfg: dict, gkeys=None, model_filter=None, eval_filter=None):
    """Ordered, de-duplicated (model, eval) cells requested across groups."""
    cells, seen = [], set()
    for gkey, g in cfg["groups"].items():
        if gkeys and gkey not in gkeys:
            continue
        for mkey in g["models"]:
            if model_filter and mkey not in model_filter:
                continue
            for ekey in g["evals"]:
                if eval_filter and ekey not in eval_filter:
                    continue
                if (mkey, ekey) not in seen:
                    seen.add((mkey, ekey))
                    cells.append((mkey, ekey))
    return cells


def result_path(cfg: dict, mkey: str, ekey: str) -> Path:
    return cfg["results_dir"] / f"{mkey}__{ekey}.json"


def looks_like_hf_id(name: str) -> bool:
    return re.fullmatch(r"[\w.-]+/[\w.-]+", name) is not None and \
        not name.split("/")[0] in {"outputs", "data", "results", "checkpoints"}


def resolve_model(path_str: str) -> str | None:
    """Return a loadable model path/id, or None if the model is not ready yet."""
    p = Path(path_str)
    if p.is_dir():
        if (p / "config.json").exists():
            return str(p)
        ckpts = sorted(p.glob("checkpoint-*"),
                       key=lambda c: int(c.name.split("-")[-1]))
        ckpts = [c for c in ckpts if (c / "config.json").exists()]
        if ckpts:
            logger.warning("%s has no final model; using latest checkpoint %s",
                           p, ckpts[-1].name)
            return str(ckpts[-1])
        logger.warning("skipping model %s: directory exists but contains no "
                       "config.json and no checkpoint-*", p)
        return None
    if looks_like_hf_id(path_str):
        return path_str  # HF hub id, resolved by vLLM (from cache when offline)
    logger.warning("skipping model %s: not found (training not finished yet?)", path_str)
    return None


def dataset_ready(eval_cfg: dict) -> bool:
    ds = eval_cfg["dataset"]
    if looks_like_hf_id(ds) and not Path(ds).exists():
        return True  # HF dataset id
    p = Path(ds)
    if p.is_file():
        return True
    if p.is_dir():
        return (p / f"{eval_cfg.get('split', 'val')}.jsonl").exists()
    return False


# ---------------------------------------------------------------------------
# run: execute the matrix via scripts/evaluate.py
# ---------------------------------------------------------------------------

def eval_command(model_path: str, eval_cfg: dict, out: Path) -> list[str]:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"),
           "--model", model_path,
           "--dataset", eval_cfg["dataset"],
           "--out", str(out)]
    for key, flag in EVAL_FLAGS:
        val = eval_cfg.get(key)
        if val is not None:
            cmd += [flag, str(val)]
    if eval_cfg.get("features"):
        cmd += ["--features", *eval_cfg["features"]]
    if eval_cfg.get("dump_samples"):
        cmd.append("--dump-samples")
    return cmd


def cmd_run(cfg: dict, args) -> int:
    for name, requested, available in (("models", args.models, cfg["models"]),
                                       ("evals", args.evals, cfg["evals"]),
                                       ("groups", args.groups, cfg["groups"])):
        unknown = set(requested or []) - set(available)
        if unknown:
            raise SystemExit(f"unknown {name} key(s) {sorted(unknown)}; "
                             f"config has {sorted(available)}")

    cells = group_cells(cfg, args.groups, args.models, args.evals)
    resolved_cache: dict[str, str | None] = {}

    def resolve(mkey):
        if mkey not in resolved_cache:
            resolved_cache[mkey] = resolve_model(cfg["models"][mkey]["path"])
        return resolved_cache[mkey]

    todo, done, not_ready = [], [], []
    for mkey, ekey in cells:
        ecfg = cfg["evals"][ekey]
        out = result_path(cfg, mkey, ekey)
        if out.exists() and not args.force:
            done.append((mkey, ekey))
            continue
        resolved = resolve(mkey)
        if resolved is None:
            not_ready.append((mkey, ekey, f"model {cfg['models'][mkey]['path']} missing"))
            continue
        if not dataset_ready(ecfg):
            not_ready.append((mkey, ekey, f"dataset {ecfg['dataset']} missing"))
            continue
        todo.append((mkey, ekey, eval_command(resolved, ecfg, out)))

    logger.info("matrix: %d cells | %d already done, %d to run, %d not ready",
                len(cells), len(done), len(todo), len(not_ready))
    for mkey, ekey, why in not_ready:
        logger.info("  not ready: %s x %s (%s)", mkey, ekey, why)

    if args.dry_run:
        for _, _, cmd in todo:
            print(shlex.join(cmd))
        return 0

    failed = []
    for i, (mkey, ekey, cmd) in enumerate(todo, 1):
        logger.info("[%d/%d] %s x %s", i, len(todo), mkey, ekey)
        logger.info("  $ %s", shlex.join(cmd))
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        if proc.returncode != 0:
            logger.error("  FAILED (exit %d) — continuing with the next cell",
                         proc.returncode)
            failed.append((mkey, ekey))
        else:
            res = json.loads(result_path(cfg, mkey, ekey).read_text())
            logger.info("  %s=%.4f on %d questions", cfg["metric"],
                        res.get(cfg["metric"], float("nan")), res["num_questions"])

    logger.info("run finished: %d ok, %d failed, %d skipped (existing), %d not ready",
                len(todo) - len(failed), len(failed), len(done), len(not_ready))
    for mkey, ekey in failed:
        logger.error("  failed: %s x %s", mkey, ekey)
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# plot: figures + summary tables from collected results
# ---------------------------------------------------------------------------
# Palette: validated reference palette from the data-viz guidelines (light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
BASELINE_GRAY = "#898781"                                  # reference series
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",      # categorical slots,
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]      # fixed order
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",         # sequential blue,
       "#256abf", "#1c5cab", "#104281", "#0d366b"]         # light -> dark
DIV = ["#a33735", "#e34948", "#f0efec", "#3987e5", "#104281"]  # red<-0->blue


def load_results(cfg: dict) -> dict[tuple[str, str], dict]:
    out = {}
    for mkey in cfg["models"]:
        for ekey in cfg["evals"]:
            p = result_path(cfg, mkey, ekey)
            if p.exists():
                out[(mkey, ekey)] = json.loads(p.read_text())
    return out


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.titlecolor": INK, "font.size": 10,
        "axes.grid": False, "grid.color": GRIDLINE, "grid.linewidth": 0.8,
    })
    return plt


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("miq_seq", SEQ)
    cmap.set_bad(GRIDLINE)
    return cmap


def div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("miq_div", DIV)
    cmap.set_bad(GRIDLINE)
    return cmap


def cell_ink(rgba) -> str:
    r, g, b = rgba[:3]
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK


def draw_heatmap(ax, values, row_labels, col_labels, cmap, vmin, vmax,
                 fmt="{:.2f}", outline=(), cell_fontsize=9):
    """Annotated heatmap with surface-colored 2px gaps; NaN cells show a dash."""
    import numpy as np
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    nrows, ncols = values.shape
    # Rotate long x labels so wide columns (long eval / model names) don't collide.
    rotate = max((len(str(c)) for c in col_labels), default=0) > 9
    ax.set_xticks(range(ncols), labels=col_labels,
                  rotation=30 if rotate else 0,
                  ha="right" if rotate else "center",
                  rotation_mode="anchor")
    ax.set_yticks(range(nrows), labels=row_labels)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(ncols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(nrows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    norm = im.norm
    for i in range(nrows):
        for j in range(ncols):
            v = values[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center",
                        color=MUTED, fontsize=cell_fontsize)
            else:
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color=cell_ink(cmap(norm(v))), fontsize=cell_fontsize)
    from matplotlib.patches import Rectangle
    for (i, j) in outline:
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=INK, linewidth=1.8))
    return im


def in_task_cells(cfg, model_keys, eval_keys):
    cells = []
    for i, mkey in enumerate(model_keys):
        mtask = cfg["models"][mkey].get("task")
        for j, ekey in enumerate(eval_keys):
            if mtask is not None and mtask == cfg["evals"][ekey].get("task"):
                cells.append((i, j))
    return cells


def save(fig, plots_dir: Path, name: str, formats):
    for ext in formats:
        path = plots_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        logger.info("wrote %s", path)


def titled(base: str, note: str) -> str:
    """Prefix a figure title with the group context line, if any."""
    return f"{note}\n{base}" if note else base


def metric_matrix(results, cfg, model_keys, eval_keys, metric):
    import numpy as np
    m = np.full((len(model_keys), len(eval_keys)), np.nan)
    for i, mkey in enumerate(model_keys):
        for j, ekey in enumerate(eval_keys):
            res = results.get((mkey, ekey))
            if res is None:
                continue
            if metric not in res:
                logger.warning("%s x %s has no metric '%s' (n=%s) — cell left empty",
                               mkey, ekey, metric, res.get("n_samples_per_question"))
                continue
            m[i, j] = res[metric]
    return m


def plot_overview_heatmap(plt, cfg, matrix, model_keys, eval_keys, metric,
                          plots_dir, formats, note=""):
    mlabels = [cfg["models"][k]["label"] for k in model_keys]
    fig, ax = plt.subplots(
        figsize=(1.6 + 1.35 * len(eval_keys), 1.3 + 0.55 * len(model_keys)),
        constrained_layout=True)
    im = draw_heatmap(ax, matrix, mlabels, eval_keys, seq_cmap(), 0.0, 1.0,
                      outline=in_task_cells(cfg, model_keys, eval_keys))
    fig.colorbar(im, ax=ax, label=metric, shrink=0.85)
    ax.set_title(titled(f"MolecularIQ {metric}: models × eval sets", note), pad=12)
    ax.set_xlabel("eval set")
    fig.text(0.01, -0.03, "outlined cell = model evaluated on its own training task",
             color=MUTED, fontsize=8)
    save(fig, plots_dir, f"heatmap_{metric}", formats)
    plt.close(fig)


def plot_delta_heatmap(plt, cfg, matrix, model_keys, eval_keys, metric,
                       plots_dir, formats, note=""):
    import numpy as np
    bkey = cfg.get("baseline")
    if bkey not in model_keys:
        logger.warning("no baseline results yet — skipping delta heatmap")
        return
    b = matrix[model_keys.index(bkey)]
    others = [k for k in model_keys if k != bkey]
    if not others:
        return
    delta = matrix[[model_keys.index(k) for k in others]] - b
    lim = np.nanmax(np.abs(delta)) if np.isfinite(delta).any() else 1.0
    lim = max(lim, 0.05)
    mlabels = [cfg["models"][k]["label"] for k in others]
    fig, ax = plt.subplots(
        figsize=(1.6 + 1.35 * len(eval_keys), 1.3 + 0.55 * len(others)),
        constrained_layout=True)
    im = draw_heatmap(ax, delta, mlabels, eval_keys, div_cmap(), -lim, lim,
                      fmt="{:+.2f}", outline=in_task_cells(cfg, others, eval_keys))
    fig.colorbar(im, ax=ax, label=f"Δ {metric} vs baseline", shrink=0.85)
    ax.set_title(titled(f"Improvement over {cfg['models'][bkey]['label']}", note),
                 pad=12)
    ax.set_xlabel("eval set")
    fig.text(0.01, -0.03, "outlined cell = model evaluated on its own training task",
             color=MUTED, fontsize=8)
    save(fig, plots_dir, "heatmap_delta_vs_baseline", formats)
    plt.close(fig)


def model_color(cfg, mkey: str) -> str:
    keys = [k for k in cfg["models"] if k != cfg.get("baseline")]
    if mkey == cfg.get("baseline"):
        return BASELINE_GRAY
    return SERIES[keys.index(mkey) % len(SERIES)]


def plot_grouped_bars(plt, cfg, matrix, model_keys, eval_keys, metric,
                      plots_dir, formats, note=""):
    import numpy as np
    fig, ax = plt.subplots(
        figsize=(2.0 + 1.9 * len(eval_keys), 4.2), constrained_layout=True)
    nm = len(model_keys)
    width = 0.8 / nm
    x = np.arange(len(eval_keys))
    for i, mkey in enumerate(model_keys):
        vals = matrix[i]
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, np.nan_to_num(vals), width=width * 0.82,
               color=model_color(cfg, mkey), label=cfg["models"][mkey]["label"],
               zorder=3)
        for p, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(p, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK_2, rotation=90 if nm > 4 else 0)
    ax.set_xticks(x, labels=eval_keys)
    ax.set_ylim(0, min(1.0, float(np.nanmax(matrix)) + 0.18)
                if np.isfinite(matrix).any() else 1.0)
    ax.set_ylabel(metric)
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_title(titled(f"MolecularIQ {metric} by eval set", note), pad=12)
    ax.legend(frameon=False, ncols=min(3, nm), loc="upper left",
              bbox_to_anchor=(0, 1.0), fontsize=9)
    save(fig, plots_dir, f"bars_{metric}", formats)
    plt.close(fig)


def plot_specialization(plt, cfg, matrix, model_keys, eval_keys, metric,
                        plots_dir, formats, note=""):
    """Per trained model: Δ vs baseline on its own task vs on the other tasks."""
    import numpy as np
    bkey = cfg.get("baseline")
    if bkey not in model_keys:
        return
    b = matrix[model_keys.index(bkey)]
    rows = []
    for mkey in model_keys:
        mtask = cfg["models"][mkey].get("task")
        if mkey == bkey or mtask is None:
            continue
        delta = matrix[model_keys.index(mkey)] - b
        own = [delta[j] for j, ekey in enumerate(eval_keys)
               if cfg["evals"][ekey].get("task") == mtask]
        other = [delta[j] for j, ekey in enumerate(eval_keys)
                 if cfg["evals"][ekey].get("task") != mtask]
        own = np.nanmean(own) if own and np.isfinite(own).any() else np.nan
        other = np.nanmean(other) if other and np.isfinite(other).any() else np.nan
        if not (np.isnan(own) and np.isnan(other)):
            rows.append((cfg["models"][mkey]["label"], own, other))
    if not rows:
        logger.warning("not enough results for the specialization plot — skipping")
        return
    labels = [r[0] for r in rows]
    own = np.array([r[1] for r in rows])
    other = np.array([r[2] for r in rows])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(2.0 + 1.6 * len(rows), 4.0),
                           constrained_layout=True)
    for pos, vals, color, label in ((x - 0.18, own, SERIES[0], "own task"),
                                    (x + 0.18, other, SERIES[1], "other tasks (mean)")):
        ax.bar(pos, np.nan_to_num(vals), width=0.32, color=color, label=label,
               zorder=3)
        for p, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(p, v + (0.008 if v >= 0 else -0.008), f"{v:+.2f}",
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=8, color=INK_2)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(x, labels=labels)
    ax.set_ylabel(f"Δ {metric} vs baseline")
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_title(titled("Specialization vs generalization (gain over baseline)",
                        note), pad=12)
    ax.legend(frameon=False, fontsize=9)
    save(fig, plots_dir, "specialization_vs_generalization", formats)
    plt.close(fig)


def ordered_bins(found: set[str]) -> list[str]:
    known = ["0-250", "250-1000", "1000+"]
    return [b for b in known if b in found] + sorted(found - set(known))


def plot_complexity(plt, cfg, results, model_keys, eval_keys, plots_dir, formats,
                    note=""):
    """Small multiples: per eval set, models x complexity bins (avg accuracy)."""
    import numpy as np
    panels = []
    for ekey in eval_keys:
        found = set()
        for mkey in model_keys:
            res = results.get((mkey, ekey))
            if res:
                found |= set(res.get("by_complexity_bin", {}))
        if found:
            panels.append((ekey, ordered_bins(found)))
    if not panels:
        logger.warning("no by_complexity_bin data — skipping complexity plot")
        return
    mlabels = [cfg["models"][k]["label"] for k in model_keys]
    ncols = min(3, len(panels))
    nrows_fig = -(-len(panels) // ncols)
    fig, axes = plt.subplots(
        nrows_fig, ncols,
        figsize=(1.9 + 1.25 * max(len(b) for _, b in panels) * ncols,
                 (1.1 + 0.5 * len(model_keys)) * nrows_fig),
        constrained_layout=True, squeeze=False)
    im = None
    for ax, (ekey, bins) in zip(axes.flat, panels):
        mat = np.full((len(model_keys), len(bins)), np.nan)
        for i, mkey in enumerate(model_keys):
            res = results.get((mkey, ekey))
            if res:
                for j, b in enumerate(bins):
                    mat[i, j] = res.get("by_complexity_bin", {}).get(b, np.nan)
        im = draw_heatmap(ax, mat, mlabels, bins, seq_cmap(), 0.0, 1.0,
                          cell_fontsize=8)
        ax.set_title(ekey, fontsize=10, color=INK_2)
        if ax not in axes[:, 0]:
            ax.set_yticks(range(len(model_keys)), labels=[""] * len(model_keys))
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axes, label="avg_accuracy", shrink=0.8)
    fig.suptitle(titled("Accuracy by molecule complexity bin (heavy-atom count)",
                        note).replace("\n", " — "), color=INK)
    save(fig, plots_dir, "heatmap_complexity", formats)
    plt.close(fig)


def plot_features(plt, cfg, results, model_keys, ekey, plots_dir, formats,
                  note=""):
    """Per eval set: construct-level (features) accuracy, features x models."""
    import numpy as np
    feats = set()
    for mkey in model_keys:
        res = results.get((mkey, ekey))
        if res:
            feats |= set(res.get("by_features", {}))
    if len(feats) < 2:
        return
    bkey = cfg.get("baseline")

    def sort_key(f):
        res = results.get((bkey, ekey)) if bkey else None
        base = (res or {}).get("by_features", {}).get(f)
        return (-(base if base is not None else -1), f)

    feats = sorted(feats, key=sort_key)
    mat = np.full((len(feats), len(model_keys)), np.nan)
    for j, mkey in enumerate(model_keys):
        by = (results.get((mkey, ekey)) or {}).get("by_features", {})
        for i, f in enumerate(feats):
            mat[i, j] = by.get(f, np.nan)
    mlabels = [cfg["models"][k]["label"] for k in model_keys]
    fig, ax = plt.subplots(
        figsize=(2.8 + 1.15 * len(model_keys), 1.6 + 0.30 * len(feats)),
        constrained_layout=True)
    im = draw_heatmap(ax, mat, feats, mlabels, seq_cmap(), 0.0, 1.0,
                      cell_fontsize=8)
    fig.colorbar(im, ax=ax, label="avg_accuracy", shrink=0.7)
    ax.set_title(titled(f"Per-construct accuracy on {ekey} "
                        "(rows sorted by baseline accuracy)", note),
                 fontsize=11, pad=12)
    save(fig, plots_dir, f"heatmap_features__{ekey}", formats)
    plt.close(fig)


def write_summary(cfg, results, matrix, model_keys, eval_keys, metric, plots_dir,
                  note=""):
    import numpy as np
    bkey = cfg.get("baseline")
    brow = matrix[model_keys.index(bkey)] if bkey in model_keys else None

    csv_path = plots_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "label", "eval", "dataset", "split", "num_questions",
                    "n", "temperature", "avg_accuracy", "pass_at_1", "pass_at_3",
                    f"delta_{metric}_vs_baseline"])
        for i, mkey in enumerate(model_keys):
            for j, ekey in enumerate(eval_keys):
                res = results.get((mkey, ekey))
                if res is None:
                    continue
                delta = ""
                if brow is not None and mkey != bkey and not np.isnan(brow[j]) \
                        and not np.isnan(matrix[i, j]):
                    delta = f"{matrix[i, j] - brow[j]:+.4f}"
                w.writerow([mkey, cfg["models"][mkey]["label"], ekey,
                            res["dataset"], res.get("split", ""),
                            res["num_questions"], res["n_samples_per_question"],
                            res["temperature"], res.get("avg_accuracy", ""),
                            res.get("pass_at_1", ""), res.get("pass_at_3", ""),
                            delta])
    logger.info("wrote %s", csv_path)

    md_path = plots_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Eval matrix — {metric}\n\n")
        if note:
            f.write(f"_{note}_\n\n")
        f.write("Cell format: `metric (Δ vs baseline)`; `–` = not evaluated yet.\n\n")
        f.write("| model | " + " | ".join(eval_keys) + " |\n")
        f.write("|---" * (len(eval_keys) + 1) + "|\n")
        for i, mkey in enumerate(model_keys):
            cells = []
            for j in range(len(eval_keys)):
                v = matrix[i, j]
                if np.isnan(v):
                    cells.append("–")
                elif brow is not None and mkey != bkey and not np.isnan(brow[j]):
                    cells.append(f"{v:.3f} ({v - brow[j]:+.3f})")
                else:
                    cells.append(f"{v:.3f}")
            f.write(f"| {cfg['models'][mkey]['label']} | " + " | ".join(cells) + " |\n")
    logger.info("wrote %s", md_path)


def plot_group(plt, cfg, results, gkey, args):
    """Render one group's figures + summary into plots_dir/<subdir>/."""
    g = cfg["groups"][gkey]
    metric = args.metric or g["metric"]
    note = g["title"]
    formats = args.formats
    model_keys = [m for m in g["models"]
                  if (not args.models or m in args.models)
                  and any((m, e) in results for e in g["evals"])]
    eval_keys = [e for e in g["evals"]
                 if (not args.evals or e in args.evals)
                 and any((m, e) in results for m in g["models"])]
    if not model_keys or not eval_keys:
        logger.warning("group '%s': no results yet — skipping", gkey)
        return False

    plots_dir = cfg["plots_dir"] / g["subdir"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info("group '%s': %d models x %d evals (metric=%s) -> %s",
                gkey, len(model_keys), len(eval_keys), metric, plots_dir)

    matrix = metric_matrix(results, cfg, model_keys, eval_keys, metric)
    plot_overview_heatmap(plt, cfg, matrix, model_keys, eval_keys, metric,
                          plots_dir, formats, note)
    plot_delta_heatmap(plt, cfg, matrix, model_keys, eval_keys, metric,
                       plots_dir, formats, note)
    plot_grouped_bars(plt, cfg, matrix, model_keys, eval_keys, metric,
                      plots_dir, formats, note)
    plot_specialization(plt, cfg, matrix, model_keys, eval_keys, metric,
                        plots_dir, formats, note)
    plot_complexity(plt, cfg, results, model_keys, eval_keys, plots_dir, formats,
                    note)
    for ekey in eval_keys:
        plot_features(plt, cfg, results, model_keys, ekey, plots_dir, formats, note)
    write_summary(cfg, results, matrix, model_keys, eval_keys, metric, plots_dir,
                  note)
    return True


def cmd_plot(cfg: dict, args) -> int:
    if args.groups:
        unknown = set(args.groups) - set(cfg["groups"])
        if unknown:
            raise SystemExit(f"unknown group(s) {sorted(unknown)}; "
                             f"config has {sorted(cfg['groups'])}")
    results = load_results(cfg)
    if not results:
        logger.error("no result files in %s — run the 'run' step first", cfg["results_dir"])
        return 1

    plt = setup_matplotlib()
    cfg["plots_dir"].mkdir(parents=True, exist_ok=True)
    gkeys = args.groups or list(cfg["groups"])
    plotted = [gkey for gkey in gkeys if plot_group(plt, cfg, results, gkey, args)]
    if not plotted:
        logger.error("nothing plotted — no group has results yet")
        return 1
    logger.info("plotted %d group(s) under %s: %s",
                len(plotted), cfg["plots_dir"], ", ".join(plotted))
    return 0


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "plot", "all"])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval_matrix.yaml")
    p.add_argument("--groups", nargs="+", default=None,
                   help="Restrict to these group keys from the config "
                        "(e.g. heldout official aromatic_ring)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Restrict to these model keys from the config")
    p.add_argument("--evals", nargs="+", default=None,
                   help="Restrict to these eval keys from the config")
    p.add_argument("--force", action="store_true",
                   help="Re-run cells even if their result file exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the evaluate.py commands instead of running them")
    p.add_argument("--metric", default=None,
                   help="Override the headline metric from the config "
                        "(avg_accuracy | pass_at_1 | pass_at_3)")
    p.add_argument("--formats", nargs="+", default=["png"],
                   help="Figure formats, e.g. --formats png pdf")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg["results_dir"].mkdir(parents=True, exist_ok=True)

    rc = 0
    if args.command in ("run", "all"):
        rc = cmd_run(cfg, args)
    if args.command in ("plot", "all") and not args.dry_run:
        rc = max(rc, cmd_plot(cfg, args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
