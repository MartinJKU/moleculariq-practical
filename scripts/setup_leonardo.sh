#!/bin/bash
# One-time environment setup on a Leonardo LOGIN node (needs internet).
# Run from the repository root:
#     bash scripts/setup_leonardo.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/env_leonardo.sh"

echo "MIQ_HOME  = $MIQ_HOME"
echo "VENV_DIR  = $VENV_DIR"
echo "HF_HOME   = $HF_HOME"

PYTHON_BIN="$(command -v python3)"
echo "Using python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
$PYTHON_BIN - <<'EOF'
import sys
assert sys.version_info >= (3, 10), f"Python >= 3.10 required, got {sys.version}"
EOF

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating venv at $VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"
pip install -e "$REPO_ROOT"

echo
echo "Sanity check..."
python - <<'EOF'
import torch, trl, transformers, datasets, vllm
import moleculariq_core
from moleculariq_grpo.rewards import score_answer
r = score_answer('<answer>{"ring_count": 0}</answer>', "count", target='{"ring_count": 0}')
assert r == 1.0, f"reward sanity check failed: {r}"
print(f"torch {torch.__version__} | trl {trl.__version__} | "
      f"transformers {transformers.__version__} | vllm {vllm.__version__} | "
      f"datasets {datasets.__version__}")
print("CUDA available:", torch.cuda.is_available(), "(False is expected on login nodes)")
print("Reward pipeline OK")
EOF

echo
echo "Setup complete. Next steps:"
echo "  1. bash scripts/download_assets.sh        # prefetch model + datasets (login node)"
echo "  2. sbatch slurm/create_datasets.slurm ... # or run scripts/create_datasets.py"
echo "  3. sbatch slurm/train.slurm configs/count.yaml"
