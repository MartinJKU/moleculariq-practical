#!/bin/bash
# Shared environment for CINECA Leonardo (login nodes AND compute nodes).
# Source this from the login shell, setup scripts and SLURM jobs:
#     source scripts/env_leonardo.sh
#
# Layout (override via env vars before sourcing):
#   MIQ_HOME  - base dir for venv + caches   (default: $WORK/$USER, else $SCRATCH)
#   VENV_DIR  - python virtualenv            (default: $MIQ_HOME/venvs/moleculariq-grpo)
#   HF_HOME   - Hugging Face cache           (default: $MIQ_HOME/hf_cache)

# --- modules ---------------------------------------------------------------
module purge 2>/dev/null
# Leonardo module names carry toolchain suffixes; fall back gracefully.
module load python/3.11.6--gcc--8.5.0 2>/dev/null \
    || module load python 2>/dev/null \
    || echo "WARNING: no python module loaded; check 'module avail python'" >&2
# No CUDA module needed: torch/vLLM pip wheels bundle the CUDA runtime
# (only the NVIDIA driver on the node is required).

# --- paths -----------------------------------------------------------------
if [ -z "${MIQ_HOME:-}" ]; then
    if [ -n "${WORK:-}" ] && [ -d "$WORK" ]; then
        MIQ_HOME="$WORK/$USER"
    elif [ -n "${CINECA_SCRATCH:-}" ]; then
        MIQ_HOME="$CINECA_SCRATCH"
    else
        MIQ_HOME="$HOME"
    fi
fi
export MIQ_HOME
export VENV_DIR="${VENV_DIR:-$MIQ_HOME/venvs/moleculariq-grpo}"
export HF_HOME="${HF_HOME:-$MIQ_HOME/hf_cache}"
mkdir -p "$HF_HOME"

# --- runtime ---------------------------------------------------------------
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

# Leonardo compute nodes have NO internet access: force offline mode inside
# SLURM jobs so transformers/datasets/vllm only read the pre-populated cache
# (see scripts/download_assets.sh).
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
fi

# --- venv ------------------------------------------------------------------
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi
