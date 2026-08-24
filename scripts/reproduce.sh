#!/usr/bin/env bash
#
# Reproduce every number and figure in the report, end to end, from a clean
# checkout. Each stage is idempotent and skipped if its output already exists,
# so the script can be re-run after an interruption without repeating work.
#
# All randomness is pinned: dataset generation uses --seed 42 (matching the
# training configs), training uses the seed in configs/*.yaml, and evaluation
# uses --seed 0 with greedy decoding for the held-out sets. Re-running on the
# same hardware and the pinned software stack (requirements.txt) reproduces the
# reported results exactly; across different GPU models, floating-point
# non-determinism in the attention kernels can move individual accuracies by a
# few tenths of a percent, which is well inside the reported confidence
# intervals.
#
# Usage (from the repo root):
#
#     ./scripts/reproduce.sh datasets     # stage 1 only (CPU, serial partition)
#     ./scripts/reproduce.sh train        # stage 2 only (submits 4 SLURM jobs)
#     ./scripts/reproduce.sh eval         # stage 3 only (the eval matrix)
#     ./scripts/reproduce.sh report       # stage 4 only (figures + tables)
#     ./scripts/reproduce.sh diagnose     # training data vs benchmark (CPU)
#     ./scripts/reproduce.sh all          # stages 1-4, in order
#
# On Leonardo, stages 1-3 need a compute allocation: run them via the SLURM
# wrappers in slurm/ (this script prints the exact sbatch commands when it
# detects it is running on a login node). The report and diagnose stages are
# CPU-only and run anywhere.
#
# The project environment (modules + venv) is loaded automatically via
# scripts/env_leonardo.sh, so this works from a bare login shell. Override the
# interpreter with PYTHON=/path/to/python if you need a specific one.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAGE="${1:-all}"
SEED=42
TRAIN_SIZE=20000
VAL_SIZE=500

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- environment -----------------------------------------------------------
# Load the project environment (modules, caches, venv) the same way the SLURM
# wrappers do, so this script works from a bare login shell with nothing
# activated. Sourced with errexit/nounset off: `module` is absent on machines
# that are not Leonardo, and a non-zero return there must not abort the run.
if [ -f "$REPO_ROOT/scripts/env_leonardo.sh" ]; then
    set +eu
    # shellcheck disable=SC1091
    source "$REPO_ROOT/scripts/env_leonardo.sh"
    set -eu
fi

# Resolve an interpreter explicitly rather than assuming `python` is on PATH --
# outside an activated venv many systems provide only `python3`.
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: no python interpreter found." >&2
    echo "  Activate the project venv first:  source scripts/env_leonardo.sh" >&2
    echo "  (or set PYTHON=/path/to/python and re-run)" >&2
    exit 1
fi
echo "Using interpreter: $(command -v "$PY")"

# --------------------------------------------------------------------------
# Stage 1: datasets
# --------------------------------------------------------------------------
stage_datasets() {
    log "Stage 1/4: generating question datasets (seed=$SEED)"
    local specs=(
        "count:data/count:"
        "index:data/index:"
        "constraint:data/constraint:"
        "count:data/count_aromatic_ring:aromatic_ring"
    )
    for spec in "${specs[@]}"; do
        IFS=: read -r task out constructs <<< "$spec"
        if [[ -f "$out/val.jsonl" ]]; then
            echo "  [skip] $out already exists"
            continue
        fi
        echo "  [run ] $out"
        # shellcheck disable=SC2086  # $constructs is intentionally word-split
        "$PY" scripts/create_datasets.py \
            --task "$task" \
            --out "$out" \
            --train-size "$TRAIN_SIZE" \
            --val-size "$VAL_SIZE" \
            --seed "$SEED" \
            ${constructs:+--constructs $constructs}
    done
}

# --------------------------------------------------------------------------
# Stage 2: training (one single-task GRPO run per config)
# --------------------------------------------------------------------------
stage_train() {
    log "Stage 2/4: GRPO training"
    for cfg in configs/count.yaml configs/index.yaml configs/constraint.yaml \
               configs/count_aromatic_ring.yaml; do
        local out
        out="$("$PY" - "$cfg" <<'PYEOF'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["output_dir"])
PYEOF
)"
        if [[ -f "$out/config.json" ]]; then
            echo "  [skip] $out already trained"
            continue
        fi
        if command -v sbatch >/dev/null 2>&1; then
            # train.slurm takes the config POSITIONALLY (it shifts $1 and
            # forwards the rest to train_grpo.py) -- passing --config here
            # would be consumed as the config path itself.
            echo "  [submit] sbatch slurm/train.slurm $cfg"
            sbatch slurm/train.slurm "$cfg"
        else
            echo "  [run ] $cfg"
            "$PY" scripts/train_grpo.py --config "$cfg"
        fi
    done
    if command -v sbatch >/dev/null 2>&1; then
        echo
        echo "  Training jobs submitted. Wait for them to finish (squeue -u \$USER),"
        echo "  then run: ./scripts/reproduce.sh eval"
    fi
}

# --------------------------------------------------------------------------
# Stage 3: evaluation matrix
# --------------------------------------------------------------------------
stage_eval() {
    log "Stage 3/4: evaluation matrix (models x eval sets)"
    if command -v sbatch >/dev/null 2>&1 && [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "  [submit] sbatch slurm/eval_matrix.slurm"
        sbatch slurm/eval_matrix.slurm
        echo "  When the job finishes, run: ./scripts/reproduce.sh report"
    else
        "$PY" scripts/compare_evals.py run --config configs/eval_matrix.yaml
    fi
}

# --------------------------------------------------------------------------
# Stage 4: figures and tables for the report
# --------------------------------------------------------------------------
stage_report() {
    log "Stage 4/4: rendering figures and summary tables"
    "$PY" scripts/compare_evals.py plot \
        --config configs/eval_matrix.yaml --formats png pdf
    echo
    echo "  Figures + summary.csv/summary.md per group under results/matrix/plots/"
    echo "  The report in report/ includes these PDFs directly."
}

# --------------------------------------------------------------------------
# Diagnostics: is training measuring the same task the benchmark measures?
# --------------------------------------------------------------------------
stage_diagnose() {
    log "Diagnostics: generated training data vs official benchmark splits"
    local specs=(
        "count:data/count/train.jsonl:single_count"
        "index:data/index/train.jsonl:single_index"
    )
    for spec in "${specs[@]}"; do
        IFS=: read -r task gen split <<< "$spec"
        if [[ ! -f "$gen" ]]; then
            echo "  [skip] $gen not found — run stage 'datasets' first"
            continue
        fi
        "$PY" scripts/compare_distributions.py \
            --task "$task" --generated "$gen" \
            --official-split "$split" --out "results/dist/$task"
    done
    echo
    echo "  Distribution comparisons in results/dist/ (figure + JSON per task)."
    echo "  A TVD above ~0.3 on any axis means the generator and the benchmark"
    echo "  pose meaningfully different tasks."
}

case "$STAGE" in
    datasets) stage_datasets ;;
    train)    stage_train ;;
    eval)     stage_eval ;;
    diagnose) stage_diagnose ;;
    report)   stage_report ;;
    all)      stage_datasets; stage_train; stage_eval; stage_report ;;
    *)        echo "usage: $0 {datasets|train|eval|diagnose|report|all}" >&2; exit 2 ;;
esac

log "Done: stage '$STAGE'"
