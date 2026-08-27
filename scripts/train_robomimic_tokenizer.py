#!/usr/bin/env python3
"""Train the 2026-04-28 OAT tokenizer recipe on robomimic actions.

The defaults reproduce the tokenizer setup from
``output/20260428/194619_train_oattok_libero10_N500`` while changing the
dataset to the local three-task robomimic zarr:

    python scripts/train_robomimic_tokenizer.py

Use ``--smoke-test`` to execute one train, validation, and reconstruction
batch before launching the full job.
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


def optional_step_override(name: str, value: int | None) -> list[str]:
    if value is None:
        return []
    return [f"training.{name}={value}"]


def get_environment_prefix(conda_env: str) -> list[str]:
    """Prefer uv, then the active environment, then the named conda env."""
    if shutil.which("uv") is not None:
        return ["uv", "run"]
    if shutil.which("accelerate") is not None:
        return []
    if shutil.which("conda") is not None:
        return ["conda", "run", "--no-capture-output", "-n", conda_env]
    raise RuntimeError(
        "Neither uv, accelerate, nor conda was found. Activate the OAT environment "
        "or install uv before launching training."
    )


def build_command(args: argparse.Namespace) -> list[str]:
    dataset_path = Path(args.dataset_path)
    if dataset_path.is_absolute():
        dataset_override = str(dataset_path)
    else:
        dataset_override = dataset_path.as_posix()

    cmd = get_environment_prefix(args.conda_env) + [
        "accelerate",
        "launch",
        "--num_machines",
        str(args.num_machines),
    ]
    if args.num_processes > 1:
        cmd.append("--multi_gpu")

    cmd.extend(
        [
            "--num_processes",
            str(args.num_processes),
            "scripts/run_workspace.py",
            "--config-name=train_oattok",
            # Use the robomimic action-only ZarrDataset configuration.
            "task/tokenizer=robomimic/mt3",
            f"task.tokenizer.dataset.zarr_path={dataset_override}",
            f"task.tokenizer.name={args.task_name}",
            f"task.tokenizer.task_name={args.task_name}",
            # Match the 2026-04-28 tokenizer run explicitly.
            "tokenizer.encoder.num_registers=8",
            "tokenizer.quantizer.levels=[8,5,5,5]",
            "training.resume=false",
            f"training.num_epochs={args.num_epochs}",
            f"training.num_demo={args.num_demo}",
            f"dataloader.batch_size={args.batch_size}",
            f"val_dataloader.batch_size={args.batch_size}",
            f"dataloader.num_workers={args.num_workers}",
            f"val_dataloader.num_workers={args.num_workers}",
            f"dataloader.persistent_workers={str(args.num_workers > 0).lower()}",
            f"val_dataloader.persistent_workers={str(args.num_workers > 0).lower()}",
            f"logging.mode={args.logging_mode}",
        ]
    )

    cmd.extend(optional_step_override("max_train_steps", args.max_train_steps))
    cmd.extend(optional_step_override("max_val_steps", args.max_val_steps))
    cmd.extend(optional_step_override("max_reconst_steps", args.max_reconst_steps))

    if args.smoke_test:
        cmd.extend(
            [
                "training.num_epochs=1",
                "training.max_train_steps=1",
                "training.max_val_steps=1",
                "training.max_reconst_steps=1",
                "training.checkpoint_every=1",
                "training.val_every=1",
                "training.sample_every=1",
                "dataloader.batch_size=8",
                "val_dataloader.batch_size=8",
            ]
        )
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the OAT8 + FSQ tokenizer recipe from the 2026-04-28 run "
            "on a robomimic zarr dataset."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Robomimic zarr dataset (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--task-name",
        default="robomimic_mt3",
        help="Name used in Hydra output directories and W&B logs.",
    )
    parser.add_argument(
        "--num-demo",
        type=int,
        default=600,
        help="Demo count used for experiment naming (default: 600).",
    )
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument(
        "--conda-env",
        default="oat-rfsq-pre",
        help="Fallback conda environment when uv is unavailable (default: oat-rfsq-pre).",
    )
    parser.add_argument("--num-epochs", type=int, default=5001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--logging-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
    )
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-steps", type=int)
    parser.add_argument("--max-reconst-steps", type=int)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one small batch for train/validation/reconstruction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without launching training.",
    )
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
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
