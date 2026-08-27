#!/usr/bin/env python3
"""Train an original FSQ-OAT tokenizer with 16 register tokens.

Default usage:
    python scripts/train_oat16_tokenizer.py

This is the tokenizer side of the OAT16 baseline. It keeps the original OAT
tokenizer architecture/quantizer, but changes the register-token count from 8
to 16 via Hydra:

    tokenizer.encoder.num_registers=16

The decoder latent horizon is tied to this value in the config, so the produced
tokenizer checkpoint should be used by the OAT16 policy training script.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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
            "--config-name=train_oattok",
            "name=train_oattok_oat16",
            "task/tokenizer=libero/libero10",
            "tokenizer.encoder.num_registers=16",
            "training.resume=false",
            f"training.num_epochs={args.num_epochs}",
            f"training.num_demo={args.num_demo}",
            f"dataloader.batch_size={args.batch_size}",
            f"val_dataloader.batch_size={args.batch_size}",
            f"dataloader.num_workers={args.num_workers}",
            f"val_dataloader.num_workers={args.num_workers}",
            "dataloader.persistent_workers=false",
            "val_dataloader.persistent_workers=false",
            f"logging.mode={args.logging_mode}",
        ]
    )
    return accelerate_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an original FSQ-OAT tokenizer with 16 register tokens."
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
        "--num-epochs",
        type=int,
        default=5001,
        help="Tokenizer training epochs. Default matches the OAT tokenizer run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Tokenizer train/val dataloader batch size. Default: 256.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Tokenizer train/val dataloader workers. Default: 0.",
    )
    parser.add_argument(
        "--num-demo",
        type=int,
        default=500,
        help="Number of LIBERO demos. Default: 500.",
    )
    parser.add_argument(
        "--logging-mode",
        default="disabled",
        help="Hydra/W&B logging mode. Default: disabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = build_command(args)
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"

    print("Running from:", REPO_ROOT)
    print("Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
