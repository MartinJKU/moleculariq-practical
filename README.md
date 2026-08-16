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
  reproduce.sh            # one entry point that reruns STEPS 1-4 with pinned seeds
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
  stats.py                # confidence intervals + paired significance tests
tests/
  test_stats.py           # unit tests (pytest -q)
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

## Reproducing everything (`scripts/reproduce.sh`)

One entry point reruns the whole pipeline with pinned seeds, skipping any stage
whose output already exists:

```bash
./scripts/reproduce.sh all        # datasets -> training -> eval matrix -> figures
./scripts/reproduce.sh datasets   # or run a single stage
./scripts/reproduce.sh report     # figures + tables only (CPU, no GPU needed)
```

On a login node with `sbatch` available, the training and evaluation stages
submit the SLURM jobs in `slurm/` instead of running inline. Randomness is
pinned throughout: datasets use `--seed 42`, training uses `seed: 42` from
`configs/*.yaml`, and evaluation uses `--seed 0` with greedy decoding on the
held-out sets. On the pinned stack in `requirements.txt` and the same GPU model
this reproduces the reported numbers exactly; across different GPUs,
floating-point non-determinism in the attention kernels moves individual
accuracies by a few tenths of a percent — well inside the reported confidence
intervals.

Run the unit tests with:

```bash
pytest -q          # 27 tests covering the statistics in src/moleculariq_grpo/stats.py
```

## Uncertainty and significance

Evaluating on a finite question sample is a measurement, so every headline
number is reported with an interval, and every comparison against the baseline
gets a paired test (`src/moleculariq_grpo/stats.py`):

| Quantity | Estimator | Why |
|---|---|---|
| `avg_accuracy` | percentile bootstrap over questions | per-question scores are fractional when `--n > 1`, so no binomial assumption holds |
| `pass_at_k` | Wilson score interval | strictly binomial per question; stays inside [0,1] near 0 and 1, where these small models live |
| Δ vs baseline | paired bootstrap | both models answer the *same* questions, so pairing removes question difficulty and is far more sensitive than comparing two intervals |

`evaluate.py` writes `avg_accuracy_ci`, `pass_at_k_ci` and the raw
`per_question_scores` into each result JSON; `compare_evals.py` draws the
intervals as error bars and marks significant differences (`p < 0.05`) with `*`
in `summary.md`/`summary.csv`.

## Control baseline (guard against degenerate policies)

A model can score well on a generative task by emitting one lucky constant
instead of reasoning. Two mechanisms make that visible rather than leaving it
for a reader to discover:

1. **Answer-diversity statistics.** Every evaluation reports the number of
   distinct answers and the share taken by the single most common one.
   `evaluate.py` logs a `DEGENERATE OUTPUT` warning above 50%, and
   `summary.md` flags those rows with ⚠️.
2. **A constant-answer control.** `--constant-answer` scores one fixed string
   against every question with no model and no GPU:

   ```bash
   python scripts/evaluate.py --constant-answer '{"smiles": "CC=O"}' \
       --dataset data/constraint/val.jsonl --out results/constraint_val_constant.json
   ```

   It is wired into `configs/eval_matrix.yaml` as the `constant` model and
   appears in every group as a hatched gray bar. **Any result that does not
   clear this floor is not evidence of task ability.** This matters most for
   constraint generation, where a small molecule satisfies a large share of the
   generated constraints — see "Known limitations" below.

## Diagnosing a weak training signal

Two instruments answer the question "why did training on task X not improve
task X on the benchmark?", both cheap and neither needing a GPU.

### 1. Is GRPO getting any gradient? (`monitoring.py`)

GRPO computes each completion's advantage *relative to its own group* — the
completions sampled for the same prompt. If every completion in a group scores
identically, `r_i - mean(r) = 0` and **that group contributes no gradient at
all**. With a binary correctness reward this is common:

- an easy prompt the model always solves → all rewards 1.0 → no gradient
- a hard prompt the model never solves → all rewards 0.0 → no gradient

so only prompts the policy is *already borderline* on actually train it. This
is invisible in the mean-reward curve: a run can show a healthy mean while most
groups are saturated and teaching nothing.

`RewardGroupMonitor` wraps the correctness reward and records the fraction of
degenerate groups. It is enabled by default (`monitor_reward_groups: true` in
the training config), is a transparent pass-through — it cannot change rewards
or training — and writes `reward_groups.jsonl` into the run's `output_dir`,
logging every 10 calls:

```
[reward groups] call 30 | no-signal 71.2% (all-wrong 58.1%, all-correct 13.1%)
                | mean group std 0.104 | mean reward 0.180
```

Read it as a diagnosis: a high **all-wrong** share means the reward is too
sparse (consider a graded reward that orders incorrect answers, so a group of
16 wrong answers still has variance); a high **all-correct** share means the
prompts are too easy and the run has saturated.

### 2. Is training the same task as the benchmark? (`compare_distributions.py`)

Our count-trained model improves on our generated held-out questions but
degrades on the official split — the signature of fitting the generator rather
than the task. This script compares the two distributions directly on CPU:

```bash
./scripts/reproduce.sh diagnose        # both count and index

# or one task:
python scripts/compare_distributions.py --task count \
    --generated data/count/train.jsonl \
    --official-split single_count --out results/dist/count
```

It reports the total variation distance (TVD, 0 = identical, 1 = disjoint) on
three axes, and writes a three-panel comparison figure plus a JSON summary:

| Axis | Why it matters |
|---|---|
| construct mix (`features`) | difficulty varies enormously by construct — counting rings is far easier than counting carbon atoms, so a different mix alone can flip a result |
| answer magnitude | counting to 3 and counting to 27 are different tasks |
| molecule size | heavy-atom count, approximated from SMILES |

A TVD above ~0.3 on any axis means the generator and the benchmark pose
meaningfully different tasks, and the generator should be rebalanced on that
axis before concluding anything about what training did or did not learn.

## Graded count reward (`count_partial_credit`)

Instrumenting a real counting run found **38.4% of GRPO groups uniformly wrong
and 0% uniformly correct**: a third of the rollout budget carried no
correctness signal. In those groups the only surviving gradient came from the
format shaping term — which is exactly what the training curve shows being
learned (format reward 0.14 → 0.96 over 20 steps, while correctness stayed
flat).

A binary reward cannot tell "off by one" from "off by twenty", yet ordering
those failures is precisely what would point the policy somewhere useful. The
graded reward supplies that ordering:

```
reward = 1.0                              verifier accepts
         λ · exp(-|pred - true| / τ)      a number was parsed but is wrong
         0.0                              nothing usable produced
```

Enable it with two keys in a training config (see `configs/count_graded.yaml`,
which is otherwise byte-identical to `configs/count.yaml`):

```yaml
count_partial_credit: 0.15   # λ — 0.0 disables (default)
count_partial_tau: 3.0       # τ — decay rate of the credit
```

```bash
sbatch slurm/train.slurm configs/count_graded.yaml
```

Three properties make this safe to turn on:

- **Exact match always wins.** Partial credit is capped at `λ·exp(-1/τ)` =
  **0.107**, an order of magnitude below the 1.0 an exact answer scores. No
  approximate answer can outrank a correct one, so the policy optimum is
  unchanged — only the gradient around it becomes informative.
- **Evaluation is untouched.** `evaluate.py` calls `score_answer`, which stays
  strictly binary, so reported numbers remain comparable with the official
  protocol. Partial credit exists only in the training reward.
- **Counting only.** Index answers are lists and constraint answers are
  molecules; neither has a meaningful scalar distance, and inventing one for
  constraint generation would risk a fresh reward-hacking surface on a task
  that already had one. Those tasks keep the binary reward.

Verified effect on a group of 16 uniformly-wrong completions: no-signal
fraction 100% → 0%, group std 0.0000 → 0.0273. Re-run the probe from the
diagnostics section against both configs to compare on real data.

## Known limitations

**The generated `constraint` val set is substantially easier than the official
benchmark and should not be read as a measure of constraint-satisfaction
ability.** `_build_constraint` in `src/moleculariq_grpo/data.py` anchors each
constraint on a real molecule's property value `v` and then loosens it with a
random operator. When `v = 0` — common for constructs such as bridgehead atoms,
E/Z double bonds and R/S stereocenters — *every* operator branch yields a
constraint that any molecule with zero of that property satisfies, and roughly
one in six becomes a vacuous `>= 0` that holds for every valid molecule. The
zero-answer cap that prevents this for count/index (`--max-zero-frac`) is not
applied to constraint tasks (`data.py:405`).

Consequence: the untrained baseline scores ≈0.54 on `constraint_val` but ≈0.06
on the official `single_constraint_generation` split — a 9× gap, while count
and index agree closely across both sources. The constant-answer control
reproduces most of the 0.54, confirming the cause. Treat the official split as
the result for constraint generation; the held-out constraint column is
retained only as a training diagnostic. Fixing this properly means extending
the zero-answer cap to constraints and rejecting vacuous constraints, then
regenerating `data/constraint` and retraining.

## Comparing baseline vs single-task models (`compare_evals.py`)

`scripts/compare_evals.py` automates the whole single-task comparison: it
evaluates the models against the eval datasets (one `evaluate.py` call per
cell) and renders the figures and tables you need for the practical. One
config, `configs/eval_matrix.yaml`, lists the `models` (baseline + each
single-task checkpoint) and the `evals`, and then `groups` decide what is
compared and plotted **together**. Each group is its own model × eval grid, so
one run answers both questions at once:

- **Specialization** — each task-trained model vs the baseline on its own task
  (the heatmap diagonal / the "own task" bars).
- **Generalization** — each task-trained model on the *other* tasks, e.g. the
  count model on index questions (the off-diagonal cells).

The shipped config defines three groups, each rendered into **its own subfolder**
of `results/matrix/plots/`:

| Group → folder | What it compares |
|---|---|
| `heldout/` | baseline + count/index/constraint models on our generated held-out val sets (greedy pass@1) |
| `official/` | the same models on the official `ml-jku/moleculariq-v0.0` splits (sampled pass@3) |
| `aromatic_ring/` | single-construct study: baseline vs the general count model vs the aromatic-ring specialist — kept out of the overall-task grid |

```bash
# On Leonardo (evaluate every group's cells on 1 GPU, then plot):
sbatch slurm/eval_matrix.slurm

# Or drive it directly:
python scripts/compare_evals.py all  --config configs/eval_matrix.yaml   # run + plot
python scripts/compare_evals.py run  --config configs/eval_matrix.yaml   # evals only (GPU)
python scripts/compare_evals.py plot --config configs/eval_matrix.yaml   # figures only (no GPU)

# Preview what would run without launching vLLM:
python scripts/compare_evals.py run --config configs/eval_matrix.yaml --dry-run
# Only one group / some models:
python scripts/compare_evals.py all --config configs/eval_matrix.yaml --groups heldout
python scripts/compare_evals.py run --config configs/eval_matrix.yaml --models count index
```

Cells are cached flat as `results/matrix/<model>__<eval>.json` and shared
across groups; **existing results are skipped**, so you can run the matrix now
with whatever models have finished and re-submit later to fill in the rest
(missing cells show as `–` in the plots). Models/datasets that don't exist yet
are reported as "not ready" and skipped, not errored, so a partially-trained
sweep still produces output.

Each group folder gets the full figure set (`--formats png pdf`):

| File | What it shows |
|---|---|
| `heatmap_<metric>.png` | models × eval sets, in-task cells outlined |
| `heatmap_delta_vs_baseline.png` | same grid as Δ vs baseline (diverging red/blue) |
| `bars_<metric>.png` | grouped bars, models side by side per eval set |
| `specialization_vs_generalization.png` | per model: gain on its own task vs mean gain on the others |
| `heatmap_complexity.png` | small multiples: accuracy per complexity bin, per eval set |
| `heatmap_features__<eval>.png` | per-construct (`features`) accuracy, models side by side |
| `summary.csv` / `summary.md` | every cell + Δ vs baseline as a table |

Each group sets its own headline `metric` (`avg_accuracy`, `pass_at_1`, or
`pass_at_3` when its evals use `n: 3`); `--metric` overrides it. The `official`
group needs the benchmark cached in `$HF_HOME` — run `download_assets.sh` on a
login node first (compute nodes are offline). Add a group, or move an eval
between groups, by editing the `groups:` block in the config.

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
