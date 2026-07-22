#!/bin/bash
#SBATCH --job-name=miq-eval-matrix
#SBATCH --account=EUHPC_D27_069        # your CINECA project account ('saldo -b'), or: sbatch -A <account> ...
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#
# Run every group's model x eval cells on ONE GPU, sequentially, then render
# each group's comparison heatmaps / bar charts / summary into its own subfolder
# of results/matrix/plots (heldout/, official/, aromatic_ring/). Each cell is a
# separate vLLM eval (the engine is (re)loaded per model), so budget walltime
# accordingly — the official/ group samples pass@3 (n=3) and is ~3x slower per
# cell than the greedy held-out group; bump --time if you enable more splits.
# Cells with an existing result file are skipped, so a re-submit only fills in
# what's new (e.g. after another training finishes).
#
# Submit FROM THE REPO ROOT:
#     sbatch slurm/eval_matrix.slurm
#     sbatch slurm/eval_matrix.slurm --config configs/eval_matrix.yaml --models count index
#
# Any extra args are forwarded to `compare_evals.py run`.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/env_leonardo.sh

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"

CONFIG="configs/eval_matrix.yaml"
# Pull a --config out of the passthrough args (so plotting uses the same one).
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--config" ]]; then CONFIG="${args[$((i + 1))]}"; fi
done

# 1) Evaluate every (model, dataset) cell that isn't already done.
python scripts/compare_evals.py run --config "$CONFIG" "$@"

# 2) Render heatmaps / bars / breakdowns + CSV & Markdown summary.
#    Plotting needs no GPU; keep it in the same job for convenience.
python scripts/compare_evals.py plot --config "$CONFIG" --formats png pdf
