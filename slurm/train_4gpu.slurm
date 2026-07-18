#!/bin/bash
#SBATCH --job-name=miq-grpo4
#SBATCH --account=EUHPC_D27_069        # your CINECA project account ('saldo -b'), or: sbatch -A <account> ...
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#
# GRPO training on a full Leonardo Booster node (4x A100 64GB), DDP with one
# colocated vLLM engine per rank. Keeps the same effective generation batch as
# the 1-GPU setup by reducing gradient accumulation to 1 (64 x 4 x 1 = 256).
#
# Submit FROM THE REPO ROOT:
#     sbatch slurm/train_4gpu.slurm configs/count.yaml
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/env_leonardo.sh

CONFIG="${1:?Usage: sbatch slurm/train_4gpu.slurm <config.yaml> [--grpo.<field> <value> ...]}"
shift

echo "Node: $(hostname) | GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"

accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    scripts/train_grpo.py --config "$CONFIG" \
    --grpo.gradient_accumulation_steps 1 \
    "$@"
