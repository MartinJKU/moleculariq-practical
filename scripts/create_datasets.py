#!/usr/bin/env python
"""Create single-task MolecularIQ question datasets from the training pool.

Always run this FIRST — training and (held-out) evaluation consume the
datasets written here.

Examples
--------
# Task-type datasets (all constructs of that task type):
python scripts/create_datasets.py --task count      --out data/count
python scripts/create_datasets.py --task index      --out data/index
python scripts/create_datasets.py --task constraint --out data/constraint

# Single-construct dataset (e.g. counting aromatic rings only):
python scripts/create_datasets.py --task count --constructs aromatic_ring \
    --out data/count_aromatic_ring

# List available constructs for a task:
python scripts/create_datasets.py --task count --list-constructs
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moleculariq_grpo.data import (  # noqa: E402
    TASKS,
    GenerationConfig,
    available_constructs,
    generate_split,
    load_pool,
    split_pool,
    write_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("create_datasets")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS, required=True,
                   help="Task type: count, index, or constraint (generation)")
    p.add_argument("--constructs", nargs="+", default=None,
                   help="Restrict to specific construct(s), e.g. 'aromatic_ring'. "
                        "Default: all constructs of the task type.")
    p.add_argument("--list-constructs", action="store_true",
                   help="Print available constructs for --task and exit")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (writes train.jsonl / val.jsonl / manifest.json)")
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--num-items", type=int, default=1,
                   help="Properties (count/index) or constraints (constraint) per "
                        "question. 1 = single_* tasks (default)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-zero-frac", type=float, default=0.25,
                   help="Max fraction of questions whose answer is 0/[] (count/index)")
    p.add_argument("--max-smiles-len", type=int, default=200,
                   help="Skip pool molecules with longer SMILES (keeps prompts short)")
    p.add_argument("--complexity-bins", nargs="+",
                   default=["0-250", "250-1000", "1000+"])
    p.add_argument("--pool-cache-dir", default=None,
                   help="HF datasets cache dir for the molecule pool")
    return p.parse_args()


def summarize(rows, name):
    feats = Counter(r["features"] for r in rows)
    bins = Counter(r["complexity_bin"] for r in rows)
    logger.info("%s: %d questions", name, len(rows))
    logger.info("  complexity bins: %s", dict(bins))
    logger.info("  top features: %s", feats.most_common(8))
    zero = 0
    for r in rows:
        if r["target"]:
            t = json.loads(r["target"])
            vals = list(t.values())
            if all((v == 0) or (isinstance(v, list) and not v) for v in vals):
                zero += 1
    if any(r["target"] for r in rows):
        logger.info("  zero-answer fraction: %.3f", zero / max(1, len(rows)))


def main():
    args = parse_args()

    if args.list_constructs:
        print(f"Available constructs for task '{args.task}':")
        for c in available_constructs(args.task):
            print(f"  {c}")
        return

    if args.out is None:
        raise SystemExit("--out is required (unless --list-constructs)")

    cfg = GenerationConfig(
        task=args.task,
        constructs=args.constructs,
        num_items=args.num_items,
        seed=args.seed,
        max_zero_fraction=args.max_zero_frac,
        max_smiles_len=args.max_smiles_len,
        complexity_bins=tuple(args.complexity_bins),
    )

    logger.info("Loading molecule pool (ml-jku/moleculariq-trainPool)...")
    pool = load_pool(cache_dir=args.pool_cache_dir)
    logger.info("Pool size: %d molecules", len(pool))

    train_mols, val_mols = split_pool(pool, cfg)
    logger.info("Molecule split: %d train / %d val (disjoint)", len(train_mols), len(val_mols))

    logger.info("Generating train questions...")
    train_rows = generate_split(train_mols, args.train_size, cfg,
                                uid_prefix="train", seed_offset=0)
    logger.info("Generating val questions...")
    val_rows = generate_split(val_mols, args.val_size, cfg,
                              uid_prefix="val", seed_offset=1)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_rows, args.out / "train.jsonl")
    write_jsonl(val_rows, args.out / "val.jsonl")

    manifest = {
        "task": args.task,
        "constructs": args.constructs or "all",
        "num_items": args.num_items,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "seed": args.seed,
        "max_zero_fraction": args.max_zero_frac,
        "max_smiles_len": args.max_smiles_len,
        "complexity_bins": list(args.complexity_bins),
        "pool": "ml-jku/moleculariq-trainPool",
        "schema": "ml-jku/moleculariq-v0.0 compatible",
    }
    with open(args.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    summarize(train_rows, "train")
    summarize(val_rows, "val")
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
