#!/bin/bash
#SBATCH --job-name=miq-eval
#SBATCH --account=EUHPC_D27_069        # your CINECA project account ('saldo -b'), or: sbatch -A <account> ...
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#
# Evaluate a model with vLLM against a specific dataset.
# Submit FROM THE REPO ROOT, passing evaluate.py arguments through:
#
#     # Held-out val questions of a generated dataset (greedy pass@1):
#     sbatch slurm/eval.slurm --model outputs/count-qwen2.5-0.5b \
#         --dataset data/count/val.jsonl --out results/count_val.json
#
#     # Official benchmark split, paper-style sampled pass@3:
#     sbatch slurm/eval.slurm --model outputs/count-qwen2.5-0.5b \
#         --dataset ml-jku/moleculariq-v0.0 --split single_count \
#         --n 3 --temperature 1.0 --out results/count_official.json
#
#     # Untrained baseline for comparison:
#     sbatch slurm/eval.slurm --model Qwen/Qwen2.5-0.5B-Instruct \
#         --dataset data/count/val.jsonl --out results/count_val_baseline.json
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/env_leonardo.sh

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"

python scripts/evaluate.py "$@"
