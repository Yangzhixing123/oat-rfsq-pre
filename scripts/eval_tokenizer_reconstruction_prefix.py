#!/usr/bin/env python3
"""Evaluate tokenizer reconstruction MSE with optional prefix token truncation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


if __name__ == "__main__":
    ROOT_DIR = str(Path(__file__).resolve().parents[1])
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute action reconstruction MSE for a tokenizer checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Tokenizer checkpoint.")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Use only the first K tokens. Omit for full-token autoencode.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device inside the visible CUDA set. Default: cuda:0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional quick-test limit on the number of validation batches.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import hydra
    import torch
    import torch.nn.functional as F
    import tqdm
    from torch.utils.data import DataLoader

    from oat.common.pytorch_util import dict_apply
    from oat.tokenizer.base_tokenizer import BaseTokenizer

    device = torch.device(args.device)
    tokenizer, cfg = BaseTokenizer.from_checkpoint(
        args.checkpoint,
        return_configuration=True,
    )
    tokenizer.to(device)
    tokenizer.eval()

    dataset = hydra.utils.instantiate(cfg.task.tokenizer.dataset)
    val_dataset = dataset.get_validation_dataset()
    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    total_mse = 0.0
    total_count = 0
    max_seq_len = getattr(tokenizer, "token_seq_len", None)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(
            tqdm.tqdm(dataloader, desc="Reconstruction", leave=False)
        ):
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            samples = batch["action"]

            if args.k is None:
                reconst_samples = tokenizer.autoencode(samples=samples)
                used_k = max_seq_len
            else:
                tokens = tokenizer.tokenize(samples)
                if args.k > tokens.shape[1]:
                    raise ValueError(
                        f"Requested k={args.k}, but tokenizer produced {tokens.shape[1]} tokens."
                    )
                reconst_samples = tokenizer.detokenize(tokens=tokens[:, : args.k])
                used_k = args.k

            mse = F.mse_loss(reconst_samples, samples).item()
            batch_size = samples.shape[0]
            total_mse += mse * batch_size
            total_count += batch_size

            if args.max_batches is not None and batch_idx >= args.max_batches - 1:
                break

    result = {
        "checkpoint": args.checkpoint,
        "k": used_k,
        "requested_k": args.k,
        "num_samples": total_count,
        "mse": total_mse / total_count,
    }

    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
