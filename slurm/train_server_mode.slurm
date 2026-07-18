#!/bin/bash
#SBATCH --job-name=miq-grpo-srv
#SBATCH --account=<ACCOUNT>        # your CINECA project account ('saldo -b'), or: sbatch -A <account> ...
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#
# OPTIONAL alternative to colocate: TRL vLLM *server* mode on a full node.
# GPU 3 runs a dedicated `trl vllm-serve` generation server, GPUs 0-2 train
# with DDP. Higher rollout throughput, but more moving parts — use
# slurm/train.slurm (colocate) unless generation is your bottleneck.
#
# Submit FROM THE REPO ROOT:
#     sbatch slurm/train_server_mode.slurm configs/count.yaml
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/env_leonardo.sh

CONFIG="${1:?Usage: sbatch slurm/train_server_mode.slurm <config.yaml> [--grpo.<field> <value> ...]}"
shift

MODEL=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG')).get('model_name_or_path','Qwen/Qwen2.5-0.5B-Instruct'))")
PORT=8765

echo "Starting vLLM server for $MODEL on GPU 3 (port $PORT)..."
CUDA_VISIBLE_DEVICES=3 trl vllm-serve \
    --model "$MODEL" \
    --port "$PORT" \
    --gpu_memory_utilization 0.9 \
    --max_model_len 2048 \
    > "logs/vllm-serve-${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# Wait for the server to come up (model load takes a while on first start)
for i in $(seq 1 120); do
    if curl -s -o /dev/null "http://localhost:${PORT}/health/"; then
        echo "vLLM server is up after ~$((i * 5))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "vLLM server died — see logs/vllm-serve-${SLURM_JOB_ID}.log" >&2
        exit 1
    fi
    sleep 5
done
curl -s -o /dev/null "http://localhost:${PORT}/health/" || {
    echo "vLLM server did not become healthy in time" >&2; exit 1; }

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
    --num_processes 3 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    scripts/train_grpo.py --config "$CONFIG" \
    --grpo.vllm_mode server \
    --grpo.vllm_server_host localhost \
    --grpo.vllm_server_port "$PORT" \
    --grpo.gradient_accumulation_steps 2 \
    --grpo.per_device_train_batch_size 48 \
    "$@"
# Effective generation batch: 48 x 3 x 2 = 288 = 18 prompts x 16 generations.
