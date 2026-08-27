#!/usr/bin/env python3
"""Train an OAT16 policy with the RFSQ policy training recipe.

Default usage:
    python scripts/train_oat16_policy_rfsq_recipe.py \
        --tokenizer-checkpoint output/.../train_oattok_oat16.../checkpoints/ep-XXXX_mse-XXXX.ckpt

This is the policy side of the OAT16 baseline. It keeps the original OAT policy
architecture, but loads a 16-register OAT tokenizer checkpoint and uses the same
policy recipe used for the RFSQ policy comparison.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_command(args: argparse.Namespace) -> list[str]:
    if args.no_uv:
        accelerate_args = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_machines",
            str(args.num_machines),
        ]
    else:
        if shutil.which("uv") is None:
            raise FileNotFoundError(
                "Could not find 'uv' in PATH. Activate the environment that has uv, "
                "or rerun this script with --no-uv from an environment that has accelerate."
            )
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
            "name=train_oatpolicy_oat16_rfsq_recipe",
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
        description="Train OAT16 policy with the RFSQ policy recipe."
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        required=True,
        help="OAT16 tokenizer checkpoint produced by train_oat16_tokenizer.py.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of accelerator processes/GPUs. Default: 1.",
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
    parser.add_argument(
        "--no-uv",
        action="store_true",
        help="Use the current Python environment instead of 'uv run'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = REPO_ROOT / args.tokenizer_checkpoint
    if not args.dry_run and not ckpt_path.exists():
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
