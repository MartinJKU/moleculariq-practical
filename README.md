# moleculariq-grpo

GRPO reinforcement learning of **Qwen/Qwen2.5-0.5B-Instruct** on
[MolecularIQ](https://github.com/ml-jku/moleculariq) tasks, using
[TRL](https://github.com/huggingface/trl) with vLLM rollouts and the official
MolecularIQ **symbolic verifier as reward**. Built to run on the CINECA
**Leonardo** cluster (Booster partition, SLURM).

- **Full fine-tuning only** — no LoRA/PEFT anywhere.
- **Single-task training**: one task type per run (`count`, `index`,
  `constraint` generation), optionally narrowed to a single construct
  (e.g. *count + aromatic rings*).
- **Datasets first**: questions are generated from the official molecule pool
  ([ml-jku/moleculariq-trainPool](https://huggingface.co/datasets/ml-jku/moleculariq-trainPool),
  1.27M molecules) into the exact schema of the official benchmark
  ([ml-jku/moleculariq-v0.0](https://huggingface.co/datasets/ml-jku/moleculariq-v0.0)),
  so the same reward/eval code path handles generated and official data.

## How it fits the MolecularIQ ecosystem

| Piece | Used here for |
|---|---|
| [`moleculariq-core`](https://github.com/ml-jku/moleculariq-core) | Question generation (`MolecularIQD`), symbolic verification (`evaluate_answer`) — the training reward |
| [`moleculariq-trainPool`](https://huggingface.co/datasets/ml-jku/moleculariq-trainPool) | Source molecules for training questions (val/test pools are hidden upstream to prevent leakage) |
| [`moleculariq-v0.0`](https://huggingface.co/datasets/ml-jku/moleculariq-v0.0) | Official benchmark for final evaluation |
| [`moleculariq-eval`](https://github.com/ml-jku/moleculariq-eval) | Answer-extraction protocol (vendored in `src/moleculariq_grpo/extraction.py`) and the system prompt |

Answers follow the official protocol: the model must output
`<answer>{...JSON...}</answer>`; extraction and scoring are identical to the
official harness (binary reward: 1.0 only if *all* requested properties /
constraints are verified correct).

## Repository layout

```
configs/                  # training configs (one per task / construct)
  eval_matrix.yaml        # + models x eval sets for scripts/compare_evals.py
scripts/
  create_datasets.py      # STEP 1: generate question datasets from the pool
  train_grpo.py           # STEP 2: GRPO training (TRL, full FT, vLLM rollouts)
  evaluate.py             # STEP 3: vLLM eval against any dataset + verifier metrics
  compare_evals.py        # STEP 4: run the model x dataset eval matrix + plot it
  env_leonardo.sh         # shared Leonardo environment (modules, caches, offline mode)
  setup_leonardo.sh       # one-time venv setup (login node)
  download_assets.sh      # prefetch model + datasets for offline compute nodes
slurm/
  create_datasets.slurm   # dataset generation on the serial partition
  train.slurm             # 1x A100, vLLM colocate (default, known-good)
  train_4gpu.slurm        # full node, DDP + colocated vLLM per rank
  train_server_mode.slurm # optional: 3 train GPUs + 1 dedicated vLLM server GPU
  eval.slurm              # evaluation job (single model x dataset)
  eval_matrix.slurm       # STEP 4: whole baseline-vs-trained comparison + plots
src/moleculariq_grpo/
  data.py                 # question generation in the official benchmark schema
  rewards.py              # TRL reward functions wrapping the official verifier
  extraction.py           # official answer extraction (vendored, MIT)
  prompts.py              # official system prompt
```

## Quickstart on Leonardo

```bash
# 0) One-time setup (LOGIN node — has internet)
bash scripts/setup_leonardo.sh          # venv in $WORK, pinned dependencies
bash scripts/download_assets.sh         # model + pool + benchmark into $HF_HOME

# The slurm/*.slurm files are set to  #SBATCH --account=EUHPC_D27_069
# (override at submit time with:  sbatch -A <account> ...)

# 1) CREATE THE DATASETS FIRST (compute nodes are offline; pool is pre-cached)
sbatch slurm/create_datasets.slurm --task count      --out data/count
sbatch slurm/create_datasets.slurm --task index      --out data/index
sbatch slurm/create_datasets.slurm --task constraint --out data/constraint
# single-construct variant (count + aromatic rings):
sbatch slurm/create_datasets.slurm --task count --constructs aromatic_ring \
    --train-size 10000 --out data/count_aromatic_ring

# 2) TRAIN (single task per run; full fine-tune; vLLM colocate)
sbatch slurm/train.slurm configs/count.yaml
sbatch slurm/train.slurm configs/index.yaml
sbatch slurm/train.slurm configs/constraint.yaml
sbatch slurm/train.slurm configs/count_aromatic_ring.yaml
# 30-minute smoke test first (recommended):
sbatch --qos=boost_qos_dbg --time=00:30:00 slurm/train.slurm \
    configs/count.yaml --grpo.max_steps 5

# 3) EVALUATE a trained model against a specific dataset
# Held-out questions from our generated dataset (greedy pass@1):
sbatch slurm/eval.slurm --model outputs/count-qwen2.5-0.5b \
    --dataset data/count/val.jsonl --out results/count_val.json
# Official benchmark split, paper-style sampled pass@3:
sbatch slurm/eval.slurm --model outputs/count-qwen2.5-0.5b \
    --dataset ml-jku/moleculariq-v0.0 --split single_count \
    --n 3 --temperature 1.0 --out results/count_official.json
# Baseline for comparison:
sbatch slurm/eval.slurm --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset data/count/val.jsonl --out results/count_val_baseline.json

# 4) COMPARE baseline vs single-task models across every task at once
#    (one GPU job runs the whole model x dataset matrix, then plots it)
sbatch slurm/eval_matrix.slurm
```

## Dataset creation details

`scripts/create_datasets.py` writes `train.jsonl` / `val.jsonl` /
`manifest.json` in the official benchmark schema (`uid`, `task_type`,
`features`, `question`, `target`, `constraints`, `original_smiles`,
`complexity_bin`, `multi_task_load`, `metadata`).

- **Molecule split**: train/val question sets use *disjoint* molecules.
- **Complexity stratification**: round-robin over the official bins
  (`0-250`, `250-1000`, `1000+`), like the benchmark.
- **Answer balancing**: `--max-zero-frac` (default 0.25) caps questions whose
  answer is `0`/`[]`, so the policy cannot reward-hack by always answering zero.
- **Prompt length control**: `--max-smiles-len` (default 200 chars) caps the
  question SMILES (TRL 1.x has no prompt truncation — length is a data concern).
- **Constraint tasks**: constraints are anchored on a real pool molecule and
  double-checked with the official verifier, so **every generated question is
  satisfiable** and verifier-supported (a molecule achieving reward 1.0 exists).
- `--constructs` restricts to specific constructs
  (`--list-constructs` shows all ~30 per task, incl. `functional_group`);
  `--num-items k` produces `multi_*` questions with k properties/constraints.

## Training details

- TRL `GRPOTrainer`, **full fine-tune** (the script refuses PEFT config).
- Rewards: `correctness_reward` (official verifier, binary) with weight 1.0
  plus a small `format_reward` (valid `<answer>{JSON}</answer>`) with weight
  0.2 to bootstrap the output format on a 0.5B model — set `format_weight: 0`
  for pure correctness.
- Rollouts via **vLLM colocate** (TRL 1.8 default): the vLLM engine shares
  each training GPU, no separate server needed. `train_server_mode.slurm`
  shows the server alternative (3 train GPUs + 1 generation GPU).
- Defaults per step: 256 completions = 16 prompts x 16 generations,
  `max_completion_length` 512, lr 1e-6, `beta` 0 (no KL / no ref model),
  TRL's default DAPO-style loss. All GRPOConfig fields can be set in the
  YAML `grpo:` block or overridden ad hoc: `--grpo.<field> <value>`.
- Checkpoints every 100 steps; reruns auto-resume from the latest
  `checkpoint-*` in `output_dir`. TensorBoard logs in `output_dir/runs`.

## Evaluation details

`scripts/evaluate.py` generates with vLLM (offline engine) and scores with the
official extraction + verifier — the same code as the training reward. It
reports `avg_accuracy`, `pass@1` (and `pass@3`/`pass@5` with `--n`), plus
per-`task_type`, per-`features` and per-`complexity_bin` breakdowns, and dumps
per-question extracted answers for inspection. It accepts our generated
JSONL datasets *and* any official `ml-jku/moleculariq-v0.0` split
(`single_count`, `multi_count`, `single_index`, `multi_index`,
`single_constraint_generation`, `multi_constraint_generation`, `test`).

For leaderboard-grade numbers on the full benchmark you can also run the
official [moleculariq-eval](https://github.com/ml-jku/moleculariq-eval)
harness against the trained checkpoint; prompts/extraction here match it.

## Comparing baseline vs single-task models (`compare_evals.py`)

`scripts/compare_evals.py` automates the whole single-task comparison: it
evaluates **every model on every eval dataset** (the full cross product, one
`evaluate.py` call per cell) and renders the figures and tables you need for
the practical. One config, `configs/eval_matrix.yaml`, lists the models
(baseline + each single-task checkpoint) and the eval sets; edit it to add or
drop rows. Because a model is scored on *all* eval sets, one run answers both
questions at once:

- **Specialization** — each task-trained model vs the baseline on its own task
  (the heatmap diagonal / the "own task" bars).
- **Generalization** — each task-trained model on the *other* tasks, e.g. the
  count model on index questions (the off-diagonal cells).

```bash
# On Leonardo (evaluate the matrix on 1 GPU, then plot):
sbatch slurm/eval_matrix.slurm

# Or drive it directly:
python scripts/compare_evals.py all  --config configs/eval_matrix.yaml   # run + plot
python scripts/compare_evals.py run  --config configs/eval_matrix.yaml   # evals only (GPU)
python scripts/compare_evals.py plot --config configs/eval_matrix.yaml   # figures only (no GPU)

# Preview what would run without launching vLLM:
python scripts/compare_evals.py run --config configs/eval_matrix.yaml --dry-run
# Only some models/evals:
python scripts/compare_evals.py run --config configs/eval_matrix.yaml --models count index
```

Cells are cached as `results/matrix/<model>__<eval>.json`; **existing results
are skipped**, so you can run the matrix now with whatever models have finished
and re-submit later to fill in the rest (missing cells show as `–` in the
plots). Models/datasets that don't exist yet are reported as "not ready" and
skipped, not errored, so a partially-trained sweep still produces output.

Figures land in `results/matrix/plots/` (`--formats png pdf`):

| File | What it shows |
|---|---|
| `heatmap_<metric>.png` | models × eval sets, in-task cells outlined |
| `heatmap_delta_vs_baseline.png` | same grid as Δ vs baseline (diverging red/blue) |
| `bars_<metric>.png` | grouped bars, models side by side per eval set |
| `specialization_vs_generalization.png` | per model: gain on its own task vs mean gain on the others |
| `heatmap_complexity.png` | small multiples: accuracy per complexity bin, per eval set |
| `heatmap_features__<eval>.png` | per-construct (`features`) accuracy, models side by side |
| `summary.csv` / `summary.md` | every cell + Δ vs baseline as a table |

The headline `metric` (default `avg_accuracy`; also `pass_at_1`, or `pass_at_3`
when evals set `n: 3`) is set in the config or with `--metric`. The commented
`official_*` blocks in the config add `ml-jku/moleculariq-v0.0` splits as extra
columns (run `download_assets.sh` on a login node first).

## Environment / pinned versions

Installed by `scripts/setup_leonardo.sh` from `requirements.txt`:

| Package | Version |
|---|---|
| trl | 1.2.0 |
| vllm | 0.11.2 |
| torch | 2.9.0 (PyPI default = CUDA 12.8) |
| transformers | 4.57.6 |
| datasets | 5.0.0 |
| accelerate | 1.14.0 |
| moleculariq-core | pinned git commit `a1b8963` |

> **Why this specific (older) stack:** Leonardo Booster imposes two hard
> constraints at once that eliminate every recent vLLM wheel:
> 1. **Driver 535.x = CUDA 12 only.** A CUDA-13 build fails to import with
>    `ImportError: libcudart.so.13: cannot open shared object file` (vLLM's `_C`
>    extension), or crashes torch init with
>    `The NVIDIA driver on your system is too old (found version 12020)`.
> 2. **RHEL 8 = glibc 2.28.** A wheel tagged `manylinux_2_31`/`_2_35` has no
>    usable build, so pip falls back to a source build → `CUDA_HOME is not set`.
>
> Every vLLM ≥ 0.20 links `libcudart.so.13`; every CUDA-12 vLLM in the 0.12–0.19
> range ships only `manylinux_2_31` wheels (glibc too new). **vLLM 0.11.2** is
> the sweet spot: its wheel is `manylinux1` (installs on any glibc) **and** it
> links `libcudart.so.12`. It pins **torch 2.9.0**, whose default PyPI wheel is
> already the CUDA-12.8 build (runs on the 535 driver via CUDA minor-version
> compatibility — needs driver ≥ 525), so no special index is required.
>
> vLLM 0.11.2 sits outside TRL 1.8's window, so TRL steps down to **1.2.0**
> (supports vLLM 0.11–0.18) and transformers to **4.57.x** (vLLM 0.11.2 needs
> `transformers<5`; TRL 1.2.0 needs `>=4.56.2`). The training/eval/reward code
> here runs unchanged on this stack — verified: all GRPOConfig keys, dataset-
> column forwarding to the reward functions, checkpointing, and the
> `trl vllm-serve` CLI. Don't bump this stack until CINECA raises the driver to
> ≥ 580 (which unlocks vLLM ≥ 0.20 + torch 2.11 + TRL 1.8).

Leonardo notes ([CINECA docs](https://docs.hpc.cineca.it/hpc/leonardo.html)):

- **Compute nodes have no internet.** `scripts/env_leonardo.sh` forces
  `HF_HUB_OFFLINE=1` inside SLURM jobs; run `download_assets.sh` on a login
  node first. W&B is set to offline mode (TensorBoard is the default logger).
- Booster: `boost_usr_prod` partition, 4x A100 64GB per node, 24h walltime
  (normal QOS); `boost_qos_dbg` for 30-minute debug runs.
- Venv + HF cache live under `$WORK/$USER` (override `MIQ_HOME`/`HF_HOME`
  before sourcing `scripts/env_leonardo.sh`).
- No CUDA module needed: the pip wheels bundle the CUDA runtime — but the
  bundled CUDA **major** version must match what the node driver supports
  (driver 535 = CUDA 12.x only; see the version-pin note above).

## Citation

If you use MolecularIQ, cite the benchmark:

```bibtex
@inproceedings{bartmann2026moleculariq,
  title={Molecular{IQ}: Characterizing Chemical Reasoning Capabilities Through Symbolic Verification on Molecular Graphs},
  author={Christoph Bartmann and Johannes Schimunek and Mykyta Ielanskyi and Philipp Seidl and G{\"u}nter Klambauer and Sohvi Luukkonen},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=RqwEzZqMFv}
}
```
