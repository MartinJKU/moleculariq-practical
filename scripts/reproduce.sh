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
#     ./scripts/reproduce.sh all          # everything, in order
#
# On Leonardo, stages 1-3 need a compute allocation: run them via the SLURM
# wrappers in slurm/ (this script prints the exact sbatch commands when it
# detects it is running on a login node). Stage 4 is CPU-only and runs anywhere.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAGE="${1:-all}"
SEED=42
TRAIN_SIZE=20000
VAL_SIZE=500

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

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
        python scripts/create_datasets.py \
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
        out="$(python - "$cfg" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["output_dir"])
PY
)"
        if [[ -f "$out/config.json" ]]; then
            echo "  [skip] $out already trained"
            continue
        fi
        if command -v sbatch >/dev/null 2>&1; then
            echo "  [submit] sbatch slurm/train.slurm --config $cfg"
            sbatch slurm/train.slurm --config "$cfg"
        else
            echo "  [run ] $cfg"
            python scripts/train_grpo.py --config "$cfg"
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
        python scripts/compare_evals.py run --config configs/eval_matrix.yaml
    fi
}

# --------------------------------------------------------------------------
# Stage 4: figures and tables for the report
# --------------------------------------------------------------------------
stage_report() {
    log "Stage 4/4: rendering figures and summary tables"
    python scripts/compare_evals.py plot \
        --config configs/eval_matrix.yaml --formats png pdf
    echo
    echo "  Figures + summary.csv/summary.md per group under results/matrix/plots/"
    echo "  The report in report/ includes these PDFs directly."
}

case "$STAGE" in
    datasets) stage_datasets ;;
    train)    stage_train ;;
    eval)     stage_eval ;;
    report)   stage_report ;;
    all)      stage_datasets; stage_train; stage_eval; stage_report ;;
    *)        echo "usage: $0 {datasets|train|eval|report|all}" >&2; exit 2 ;;
esac

log "Done: stage '$STAGE'"
