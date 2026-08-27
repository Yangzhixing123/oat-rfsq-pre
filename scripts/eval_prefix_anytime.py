#!/usr/bin/env python3
"""Run prefix/anytime policy evaluation for multiple token counts.

Default RFSQ usage:
    python scripts/eval_prefix_anytime.py --k 8

Original OAT usage:
    python scripts/eval_prefix_anytime.py --method oat --checkpoint path/to/oat.ckpt --k 8

The script runs scripts/eval_policy_sim.py with --use_k_tokens K and writes the
result into a K-specific output directory. Use --ks only when you explicitly
want to run multiple K values in sequence.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RFSQ_CKPT = (
    "output/20260711/152906_train_oatpolicy_libero10_N500/"
    "checkpoints/ep-0250_sr-0.618.ckpt"
)
DEFAULT_OAT_CKPT = (
    "output/20260429/121230_train_oatpolicy_libero10_N500/"
    "checkpoints/ep-0400_sr-0.548.ckpt"
)
DEFAULT_KS = {
    "rfsq": [1, 2, 4, 8, 12, 16],
    "oat": [1, 2, 4, 8],
}


def parse_k_values(raw: str | None, method: str) -> list[int]:
    if raw is None:
        return []
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("--ks must contain at least one integer.")
    return values


def default_checkpoint(method: str) -> str:
    return DEFAULT_RFSQ_CKPT if method == "rfsq" else DEFAULT_OAT_CKPT


def default_tag(method: str, checkpoint: str) -> str:
    ckpt_name = Path(checkpoint).stem.replace("-", "").replace(".", "")
    if "ep0250" in ckpt_name or "ep_0250" in ckpt_name:
        return f"{method}_ep0250"
    if "ep0300" in ckpt_name or "ep_0300" in ckpt_name:
        return f"{method}_ep0300"
    if "ep0400" in ckpt_name or "ep_0400" in ckpt_name:
        return f"{method}_ep0400"
    return f"{method}_{ckpt_name}"


def build_command(args: argparse.Namespace, k: int, output_dir: Path) -> list[str]:
    if args.no_uv:
        cmd = [sys.executable, "scripts/eval_policy_sim.py"]
    else:
        cmd = ["uv", "run", "scripts/eval_policy_sim.py"]

    cmd.extend(
        [
            "--checkpoint",
            args.checkpoint,
            "--output_dir",
            str(output_dir),
            "--num_exp",
            str(args.num_exp),
            "--device",
            args.device,
            "--use_k_tokens",
            str(k),
        ]
    )

    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.topk is not None:
        cmd.extend(["--topk", str(args.topk)])

    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OAT/RFSQ prefix decoding for multiple K values."
    )
    parser.add_argument(
        "--method",
        choices=["rfsq", "oat"],
        default="rfsq",
        help="Select default checkpoint and K list. Default: rfsq.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Policy checkpoint to evaluate. Defaults depend on --method.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Single K value to evaluate, e.g. --k 8.",
    )
    parser.add_argument(
        "--ks",
        default=None,
        help="Comma-separated K values for an explicit batch run, e.g. 1,2,4,8.",
    )
    parser.add_argument(
        "--num-exp",
        type=int,
        default=5,
        help="Number of repeated eval runs per K. Use 10 for final tables.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device inside the visible CUDA set. Default: cuda:0.",
    )
    parser.add_argument(
        "--output-root",
        default="output/eval_prefix",
        help="Root directory for per-K eval outputs.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Output name prefix. Default is derived from method and checkpoint.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional policy sampling temperature override.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Optional policy top-k override.",
    )
    parser.add_argument(
        "--no-uv",
        action="store_true",
        help="Run with the current Python instead of 'uv run'.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Try to rerun even if an output directory already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint is None:
        args.checkpoint = default_checkpoint(args.method)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.k is not None and args.ks is not None:
        raise ValueError("Use either --k for one run or --ks for a batch run, not both.")
    if args.k is not None:
        ks = [args.k]
    else:
        ks = parse_k_values(args.ks, args.method)
    if not ks:
        raise ValueError(
            "Please pass --k for one evaluation, or --ks for an explicit batch. "
            f"Suggested {args.method} K values: {DEFAULT_KS[args.method]}"
        )
    tag = args.tag or default_tag(args.method, args.checkpoint)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["MUJOCO_GL"] = "egl"

    print("Running from:", REPO_ROOT)
    print("Checkpoint:", args.checkpoint)
    print("K values:", ",".join(str(k) for k in ks))
    print("num_exp:", args.num_exp)

    for k in ks:
        output_dir = output_root / f"{tag}_k{k}_exp{args.num_exp}"
        if output_dir.exists() and not args.rerun_existing:
            print(f"Skip existing output: {output_dir}")
            continue

        cmd = build_command(args, k, output_dir)
        print("\nK =", k)
        print("Output:", output_dir)
        print("Command:")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
