# --------------------------------------------------------------------
# Copyright (C) 2024-2025 EPFL & Apple Inc.
# Licensed under the EPFL-Apple Sample Code License (Non-Commercial)
# --------------------------------------------------------------------

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from oat.tokenizer.oat.quantizer.fsq import FSQ


__all__ = ["RFSQ"]


class RFSQ(nn.Module):
    """Two-stage residual FSQ with learned LayerNorm-style residual conditioning.

    Tokens use stage-major order:
        z1_s1, ..., zH_s1, z1_s2, ..., zH_s2

    The token codebook is shared in size across stages, so the policy can keep a
    single autoregressive vocabulary and distinguish stages by token position.
    """

    num_stages = 2

    def __init__(
        self,
        levels: List[int],
        stage2_levels: Optional[List[int]] = None,
        latent_horizon: Optional[int] = None,
        drop_quant_p: float = 0.0,
        corrupt_tokens_p: float = 0.0,
        min_corrupt_tokens_p: Optional[float] = None,
        apply_corrupt_tokens_p: float = 0.2,
        packed_call: bool = True,
        sigma_min: float = 1e-4,
        sigma_max: float = 100.0,
    ):
        super().__init__()

        stage2_levels = stage2_levels or levels
        self.stage1 = FSQ(
            levels=levels,
            drop_quant_p=drop_quant_p,
            corrupt_tokens_p=corrupt_tokens_p,
            min_corrupt_tokens_p=min_corrupt_tokens_p,
            apply_corrupt_tokens_p=apply_corrupt_tokens_p,
            packed_call=packed_call,
        )
        self.stage2 = FSQ(
            levels=stage2_levels,
            drop_quant_p=drop_quant_p,
            corrupt_tokens_p=corrupt_tokens_p,
            min_corrupt_tokens_p=min_corrupt_tokens_p,
            apply_corrupt_tokens_p=apply_corrupt_tokens_p,
            packed_call=packed_call,
        )
        if self.stage1.dim != self.stage2.dim:
            raise ValueError(
                f"RFSQ stages must have the same dim, got {self.stage1.dim} and {self.stage2.dim}."
            )
        if self.stage1.codebook_size != self.stage2.codebook_size:
            raise ValueError(
                "RFSQ requires equal stage codebook sizes for a shared policy vocabulary."
            )

        self.dim = self.stage1.dim
        self.codebook_size = self.stage1.codebook_size
        self.latent_horizon = latent_horizon
        self.token_seq_len = (
            self.num_stages * latent_horizon if latent_horizon is not None else None
        )

        self.mu2 = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.log_sigma2 = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @property
    def levels(self) -> List[int]:
        return self.stage1._levels.tolist()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  levels={self.stage1._levels.tolist()!r},\n"
            f"  stage2_levels={self.stage2._levels.tolist()!r},\n"
            f"  codebook_size={self.codebook_size!r},\n"
            f"  latent_horizon={self.latent_horizon!r},\n"
            f")"
        )

    def sigma2(self) -> Tensor:
        return self.log_sigma2.exp().clamp(min=self.sigma_min, max=self.sigma_max)

    def _infer_horizon(self, token_len: Optional[int] = None) -> int:
        if self.latent_horizon is not None:
            return self.latent_horizon
        if token_len is None:
            raise ValueError("RFSQ needs latent_horizon when token length is unknown.")
        if token_len % self.num_stages != 0:
            raise ValueError(
                "RFSQ full token sequences must be stage-major with length divisible by 2 "
                "when latent_horizon is not configured."
            )
        return token_len // self.num_stages

    def _pad_tokens(self, indices: Tensor, target_len: int) -> Tensor:
        token_len = indices.shape[-1]
        if token_len > target_len:
            raise ValueError(f"Expected at most {target_len} RFSQ tokens, got {token_len}.")
        if token_len == target_len:
            return indices

        pad = torch.zeros(
            *indices.shape[:-1],
            target_len - token_len,
            dtype=indices.dtype,
            device=indices.device,
        )
        return torch.cat([indices, pad], dim=-1)

    def _availability_mask(
        self,
        token_lens: Optional[List[int]],
        batch_size: int,
        horizon: int,
        token_len: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        if token_lens is None:
            lens = torch.full((batch_size,), token_len, dtype=torch.long, device=device)
        else:
            lens = torch.as_tensor(token_lens, dtype=torch.long, device=device)
            if lens.numel() != batch_size:
                raise ValueError(
                    f"Expected {batch_size} token lengths, got {lens.numel()}."
                )

        positions = torch.arange(horizon, device=device).unsqueeze(0)
        stage1_mask = positions < lens.unsqueeze(1)
        stage2_mask = (positions + horizon) < lens.unsqueeze(1)
        return stage1_mask.unsqueeze(-1), stage2_mask.unsqueeze(-1)

    @torch.compiler.disable
    def forward(self, latents: Tensor) -> Tuple[Tensor, torch.LongTensor]:
        if isinstance(latents, list):
            raise TypeError("RFSQ currently expects a tensor of shape [B, H_l, D_l].")
        if latents.shape[-1] != self.dim:
            raise ValueError(
                f"expected dimension of {self.dim} but found dimension of {latents.shape[-1]}"
            )
        if self.latent_horizon is not None and latents.shape[-2] != self.latent_horizon:
            raise ValueError(
                f"expected latent horizon {self.latent_horizon} but found {latents.shape[-2]}"
            )

        q1, idx1 = self.stage1(latents)
        r1 = latents.float() - q1

        sigma2 = self.sigma2()
        r1_norm = (r1 - self.mu2) / sigma2
        q2_norm, idx2 = self.stage2(r1_norm)
        q2 = q2_norm * sigma2 + self.mu2

        z_hat = q1 + q2
        tokens = torch.cat([idx1, idx2], dim=-1)
        return z_hat, tokens

    def indices_to_embedding(
        self,
        indices: Tensor,
        token_lens: Optional[List[int]] = None,
    ) -> Tensor:
        token_len = indices.shape[-1]
        horizon = self._infer_horizon(token_len)
        full_len = self.num_stages * horizon
        padded = self._pad_tokens(indices, full_len)

        idx1 = padded[..., :horizon]
        idx2 = padded[..., horizon:]
        q1 = self.stage1.indices_to_embedding(idx1)
        q2_norm = self.stage2.indices_to_embedding(idx2)
        q2 = q2_norm * self.sigma2() + self.mu2

        if padded.ndim != 2:
            raise ValueError("RFSQ detokenization expects token tensors of shape [B, T].")
        stage1_mask, stage2_mask = self._availability_mask(
            token_lens=token_lens,
            batch_size=padded.shape[0],
            horizon=horizon,
            token_len=token_len,
            device=padded.device,
        )
        q1 = torch.where(stage1_mask, q1, torch.zeros_like(q1))
        q2 = torch.where(stage2_mask, q2, torch.zeros_like(q2))

        return q1 + q2
