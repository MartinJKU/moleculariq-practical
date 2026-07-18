#!/bin/bash
# Prefetch every remote asset into $HF_HOME on a Leonardo LOGIN node (internet).
# Compute nodes run fully offline against this cache.
# Run from the repository root:
#     bash scripts/download_assets.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/env_leonardo.sh"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

echo "HF cache: $HF_HOME"

echo "==> Downloading model: $MODEL"
hf download "$MODEL" >/dev/null || huggingface-cli download "$MODEL" >/dev/null

echo "==> Downloading molecule pool: ml-jku/moleculariq-trainPool"
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("ml-jku/moleculariq-trainPool", split="train")
print(f"    cached {len(ds)} molecules")
EOF

echo "==> Downloading official benchmark: ml-jku/moleculariq-v0.0"
python - <<'EOF'
from datasets import load_dataset
for split in ["test", "single_count", "multi_count", "single_index", "multi_index",
              "single_constraint_generation", "multi_constraint_generation"]:
    ds = load_dataset("ml-jku/moleculariq-v0.0", split=split)
    print(f"    cached split {split}: {len(ds)} rows")
EOF

echo "All assets cached in $HF_HOME — compute jobs can now run offline."
