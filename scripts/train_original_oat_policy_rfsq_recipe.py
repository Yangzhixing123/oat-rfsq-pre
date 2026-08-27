#!/usr/bin/env python3
"""Train the original FSQ-OAT policy with the RFSQ policy training recipe.

Default usage:
    python scripts/train_original_oat_policy_rfsq_recipe.py

This is a fair-control run for comparing against the RFSQ policy:
same LIBERO-10 dataset, policy schedule, batch/accumulation, EMA setting,
rollout cadence, and eval runner settings as the RFSQ policy run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TOKENIZER_CKPT = (
    "output/20260428/194619_train_oattok_libero10_N500/"
    "checkpoints/ep-1000_mse-0.002.ckpt"
)


def build_command(args: argparse.Namespace) -> list[str]:
    accelerate_args = [
        "uv",
        "run",
        "accelerate",
        "launch",
        "--num_machines",
        str(args.num_machines),
    ]

    if args.num_processes > 1:
        accelerate_args.append("--multi_gpu")

    accelerate_args.extend(
        [
            "--num_processes",
            str(args.num_processes),
            "scripts/run_workspace.py",
            "--config-name=train_oatpolicy",
            "name=train_oatpolicy_fsq_rfsq_recipe",
            "task/policy=libero/libero10",
            f"policy.action_tokenizer.checkpoint={args.tokenizer_checkpoint}",
            "task.policy.lazy_eval=false",
            "task.policy.env_runner.n_parallel_envs=4",
            "training.resume=false",
            "training.use_ema=false",
            "training.num_epochs=500",
            "training.num_demo=500",
            "training.rollout_every=50",
            "training.rollout_start_epoch=99",
            "training.checkpoint_every=50",
            "dataloader.batch_size=128",
            "val_dataloader.batch_size=128",
            "training.gradient_accumulate_every=2",
            "dataloader.num_workers=0",
            "val_dataloader.num_workers=0",
            "dataloader.persistent_workers=false",
            "val_dataloader.persistent_workers=false",
            "logging.mode=disabled",
        ]
    )
    return accelerate_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train original FSQ-OAT policy with the RFSQ policy recipe."
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        default=DEFAULT_TOKENIZER_CKPT,
        help="Original FSQ-OAT tokenizer checkpoint to use.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of accelerator processes/GPUs. Default: 4.",
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        default=1,
        help="Number of machines for accelerate launch. Default: 1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = REPO_ROOT / args.tokenizer_checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {ckpt_path}")

    cmd = build_command(args)
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["MUJOCO_GL"] = "egl"

    print("Running from:", REPO_ROOT)
    print("Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
