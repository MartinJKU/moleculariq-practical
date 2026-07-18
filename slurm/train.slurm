#!/bin/bash
#SBATCH --job-name=miq-grpo
#SBATCH --account=<ACCOUNT>        # your CINECA project account ('saldo -b'), or: sbatch -A <account> ...
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#
# GRPO training on 1x A100 64GB, vLLM colocate mode (rollouts + training on
# the same GPU). This is the default, known-good setup for Qwen2.5-0.5B.
#
# Submit FROM THE REPO ROOT:
#     sbatch slurm/train.slurm configs/count.yaml
#     sbatch slurm/train.slurm configs/count_aromatic_ring.yaml
#
# Quick debug run (30 min QOS):
#     sbatch --qos=boost_qos_dbg --time=00:30:00 slurm/train.slurm \
#         configs/count.yaml --grpo.max_steps 5
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/env_leonardo.sh

CONFIG="${1:?Usage: sbatch slurm/train.slurm <config.yaml> [--grpo.<field> <value> ...]}"
shift

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "Config: $CONFIG | Extra args: $*"

python scripts/train_grpo.py --config "$CONFIG" "$@"
