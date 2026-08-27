#!/usr/bin/env python3
"""Benchmark policy.predict_action latency for a given prefix token count."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


if __name__ == "__main__":
    ROOT_DIR = str(Path(__file__).resolve().parents[1])
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure policy inference latency for one K value."
    )
    parser.add_argument("--checkpoint", required=True, help="Policy checkpoint.")
    parser.add_argument("--k", type=int, required=True, help="Number of action tokens.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device inside the visible CUDA set. Default: cuda:0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for timing. Use 1 for per-policy-call latency.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    idx = round((len(values) - 1) * q)
    return sorted(values)[idx]


def main() -> None:
    args = parse_args()

    import torch
    from oat.policy.base_policy import BasePolicy

    device = torch.device(args.device)

    with torch.inference_mode():
        policy = BasePolicy.from_checkpoint(args.checkpoint)
        policy.to(device)
        policy.eval()

        obs = policy.create_dummy_observation(batch_size=args.batch_size, device=device)
        obs = {key: value.to(device=device, dtype=policy.dtype) for key, value in obs.items()}

        for _ in range(args.warmup):
            policy.predict_action(obs, use_k_tokens=args.k)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times_ms = []
    with torch.inference_mode():
        for _ in range(args.iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            policy.predict_action(obs, use_k_tokens=args.k)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    mean = sum(times_ms) / len(times_ms)
    var = sum((x - mean) ** 2 for x in times_ms) / max(len(times_ms) - 1, 1)
    result = {
        "checkpoint": args.checkpoint,
        "k": args.k,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "device": str(device),
        "latency_ms_mean": mean,
        "latency_ms_std": var**0.5,
        "latency_ms_p50": percentile(times_ms, 0.50),
        "latency_ms_p90": percentile(times_ms, 0.90),
    }

    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
