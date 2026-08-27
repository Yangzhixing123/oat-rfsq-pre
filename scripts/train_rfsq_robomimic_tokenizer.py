#!/usr/bin/env python3
"""Train the RFSQ-OAT tokenizer on the robomimic MT3 action dataset.

The defaults reproduce the command used for:
``output/20260818/192714_train_rfsq_oattok_rm-mt3_N600``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("data/robomimic/mt3_N600.zarr")


def environment_prefix(conda_env: str) -> list[str]:
    if os.environ.get("CONDA_DEFAULT_ENV") == conda_env:
        if shutil.which("accelerate") is not None:
            return []
    if shutil.which("conda") is not None:
        return ["conda", "run", "--no-capture-output", "-n", conda_env]
    if shutil.which("accelerate") is not None:
        return []
    raise RuntimeError(
        "Neither conda nor accelerate was found. Activate the oat-rfsq-pre environment "
        "before launching training."
    )


def build_command(args: argparse.Namespace) -> list[str]:
    dataset_path = Path(args.dataset_path)
    dataset_override = str(dataset_path) if dataset_path.is_absolute() else dataset_path.as_posix()

    cmd = environment_prefix(args.conda_env) + [
        "accelerate",
        "launch",
        "--num_machines",
        str(args.num_machines),
    ]
    if args.num_processes > 1:
        cmd.append("--multi_gpu")

    num_epochs = 1 if args.smoke_test else args.num_epochs
    batch_size = 8 if args.smoke_test else args.batch_size
    cmd.extend(
        [
            "--num_processes",
            str(args.num_processes),
            "scripts/run_workspace.py",
            "--config-name=train_rfsq_oattok",
            "task/tokenizer=robomimic/mt3",
            f"task.tokenizer.dataset.zarr_path={dataset_override}",
            f"training.num_demo={args.num_demo}",
            "training.resume=false",
            f"training.num_epochs={num_epochs}",
            f"dataloader.batch_size={batch_size}",
            f"val_dataloader.batch_size={batch_size}",
            f"dataloader.num_workers={args.num_workers}",
            f"val_dataloader.num_workers={args.num_workers}",
            f"dataloader.persistent_workers={str(args.num_workers > 0).lower()}",
            f"val_dataloader.persistent_workers={str(args.num_workers > 0).lower()}",
            f"logging.mode={args.logging_mode}",
        ]
    )
    if args.smoke_test:
        cmd.extend(
            [
                "training.max_train_steps=1",
                "training.max_val_steps=1",
                "training.max_reconst_steps=1",
                "training.checkpoint_every=1",
                "training.val_every=1",
                "training.sample_every=1",
            ]
        )
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RFSQ-OAT tokenizer on a robomimic zarr dataset."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--num-demo", type=int, default=600)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=5001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--conda-env", default="oat-rfsq-pre")
    parser.add_argument(
        "--devices",
        default="0,1,2,3",
        help="CUDA_VISIBLE_DEVICES value; pass an empty string to leave it unchanged.",
    )
    parser.add_argument(
        "--logging-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path
    resolved_dataset = dataset_path if dataset_path.is_absolute() else REPO_ROOT / dataset_path
    if not resolved_dataset.is_dir():
        raise FileNotFoundError(f"Robomimic zarr dataset not found: {resolved_dataset}")

    cmd = build_command(args)
    print(f"Working directory: {REPO_ROOT}")
    print(f"Dataset: {resolved_dataset}")
    print(f"Command: {shlex.join(cmd)}")
    if args.dry_run:
        return

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
