#!/bin/bash
#SBATCH --job-name=miq-data
#SBATCH --account=EUHPC_D27_069          # your CINECA project account ('saldo -b'), or: sbatch -A <account> ...
#SBATCH --partition=lrd_all_serial   # serial partition: CPU-only, fine for dataset generation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#
# Generate a question dataset on a compute node (pool must be pre-downloaded
# with scripts/download_assets.sh). Submit FROM THE REPO ROOT, passing
# create_datasets.py arguments through, e.g.:
#     sbatch slurm/create_datasets.slurm --task count --out data/count
#     sbatch slurm/create_datasets.slurm --task count --constructs aromatic_ring \
#         --train-size 10000 --out data/count_aromatic_ring
#
# For small datasets this also runs directly on a login node:
#     python scripts/create_datasets.py --task count --out data/count
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/env_leonardo.sh

python scripts/create_datasets.py "$@"
