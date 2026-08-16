#!/usr/bin/env python
"""GRPO training of Qwen2.5-0.5B-Instruct on MolecularIQ tasks (TRL).

Full fine-tuning (no LoRA/PEFT). Rollouts via vLLM (colocate mode by default;
server mode supported through the config). The reward is the official
MolecularIQ symbolic verifier.

Usage
-----
Single GPU:
    python scripts/train_grpo.py --config configs/count.yaml

Multi-GPU (one vLLM engine per rank, colocate mode):
    accelerate launch --num_processes 4 scripts/train_grpo.py --config configs/count.yaml

Any GRPOConfig field can be overridden from the CLI, e.g.:
    python scripts/train_grpo.py --config configs/count.yaml \
        --grpo.max_steps 50 --grpo.use_vllm false --output_dir outputs/debug
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moleculariq_grpo import rewards as reward_module  # noqa: E402
from moleculariq_grpo.data import read_jsonl  # noqa: E402
from moleculariq_grpo.prompts import SYSTEM_PROMPT, build_prompt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_grpo")

# Columns forwarded to the reward functions by GRPOTrainer
REWARD_COLUMNS = ["task_type", "target", "constraints"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="YAML training config")
    p.add_argument("--output_dir", default=None, help="Override output_dir from config")
    # Collect unknown args of the form --grpo.<field> <value> as GRPOConfig overrides
    args, unknown = p.parse_known_args()
    overrides = {}
    i = 0
    while i < len(unknown):
        key = unknown[i]
        if not key.startswith("--grpo."):
            raise SystemExit(f"Unrecognized argument: {key}")
        if i + 1 >= len(unknown):
            raise SystemExit(f"Missing value for {key}")
        overrides[key[len("--grpo."):]] = yaml.safe_load(unknown[i + 1])
        i += 2
    return args, overrides


def build_dataset(dataset_dir: Path, split: str, system_prompt: str):
    from datasets import Dataset

    rows = read_jsonl(dataset_dir / f"{split}.jsonl")
    records = []
    for r in rows:
        records.append({
            "prompt": build_prompt(r["question"], system_prompt),
            "task_type": r["task_type"],
            "target": r.get("target"),
            "constraints": r.get("constraints"),
        })
    return Dataset.from_list(records)


def main():
    args, cli_overrides = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if "peft" in cfg or "lora" in cfg:
        raise SystemExit("This project does full fine-tuning only — remove peft/lora config.")

    model_name = cfg.get("model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
    dataset_dir = Path(cfg["dataset_dir"])
    system_prompt = cfg.get("system_prompt", SYSTEM_PROMPT)

    grpo_kwargs = dict(cfg.get("grpo", {}))
    grpo_kwargs.update(cli_overrides)
    if args.output_dir:
        grpo_kwargs["output_dir"] = args.output_dir
    grpo_kwargs.setdefault("output_dir", cfg.get("output_dir", "outputs/run"))

    # Reward setup: correctness is mandatory, format shaping optional.
    correctness = reward_module.correctness_reward
    # Optional instrumentation: records how many GRPO groups produce no
    # gradient (every completion scored identically). Pass-through, so it
    # cannot change the rewards or the training result.
    if cfg.get("monitor_reward_groups", True):
        from moleculariq_grpo.monitoring import RewardGroupMonitor
        correctness = RewardGroupMonitor(
            correctness,
            log_every=int(cfg.get("monitor_log_every", 10)),
            out_path=Path(grpo_kwargs["output_dir"]) / "reward_groups.jsonl",
        )
    reward_funcs = [correctness]
    reward_weights = [float(cfg.get("correctness_weight", 1.0))]
    format_weight = float(cfg.get("format_weight", 0.0))
    if format_weight > 0:
        reward_funcs.append(reward_module.format_reward)
        reward_weights.append(format_weight)
    grpo_kwargs.setdefault("reward_weights", reward_weights)

    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(**grpo_kwargs)

    train_dataset = build_dataset(dataset_dir, cfg.get("train_split", "train"), system_prompt)
    eval_dataset = None
    if grpo_kwargs.get("eval_strategy", "no") != "no":
        eval_dataset = build_dataset(dataset_dir, cfg.get("val_split", "val"), system_prompt)

    logger.info("Model: %s", model_name)
    logger.info("Train dataset: %s (%d prompts)", dataset_dir, len(train_dataset))
    logger.info("Rewards: %s (weights %s)",
                [f.__name__ for f in reward_funcs], grpo_config.reward_weights)
    logger.info("vLLM: use_vllm=%s mode=%s", grpo_config.use_vllm,
                getattr(grpo_config, "vllm_mode", None))

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # peft_config deliberately NOT set: full fine-tune only.
    )

    resume = cfg.get("resume_from_checkpoint", "auto")
    if resume == "auto":
        ckpts = sorted(Path(grpo_config.output_dir).glob("checkpoint-*"))
        resume = str(ckpts[-1]) if ckpts else None
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(grpo_config.output_dir)
    if trainer.accelerator.is_main_process:
        trainer.processing_class.save_pretrained(grpo_config.output_dir)
        with open(Path(grpo_config.output_dir) / "train_config.json", "w") as f:
            json.dump({"config_file": str(args.config), "config": cfg,
                       "cli_overrides": cli_overrides}, f, indent=2, default=str)
    logger.info("Done. Final model saved to %s", grpo_config.output_dir)


if __name__ == "__main__":
    main()
