"""OAT policy variant with paired/interleaved two-stage RFSQ tokens.

The underlying RFSQ tokenizer keeps its original stage-major serialization::

    q1_1, ..., q1_H, q2_1, ..., q2_H

This policy presents the same tokens to the autoregressive model in paired order::

    q1_1, q2_1, q1_2, q2_2, ..., q1_H, q2_H

Consequently, ``use_k_tokens=K`` means K complete RFSQ latent positions.  The
policy generates 2*K discrete tokens and the tokenizer decoder receives the K
complete ``q1_i + q2_i`` latents followed by masked latent positions.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from oat.policy.oatpolicy import OATPolicy


class RFSQPairOATPolicy(OATPolicy):
    """OATPolicy whose autoregressive unit is a complete RFSQ pair."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        quantizer = self.action_tokenizer.quantizer
        if getattr(quantizer, "num_stages", None) != 2:
            raise ValueError(
                "RFSQPairOATPolicy requires a two-stage RFSQ action tokenizer."
            )

        pair_horizon = getattr(quantizer, "latent_horizon", None)
        if pair_horizon is None:
            pair_horizon = self.action_tokenizer.latent_horizon
        if self.max_seq_len != 2 * pair_horizon:
            raise ValueError(
                "Expected two RFSQ tokens per latent position, but got "
                f"token_seq_len={self.max_seq_len} and latent_horizon={pair_horizon}."
            )

        self.pair_horizon = pair_horizon

    def _stage_major_to_interleaved(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert [all q1, all q2] into [q1_1, q2_1, q1_2, q2_2, ...]."""
        if tokens.ndim != 2:
            raise ValueError(f"Expected token tensor [B, T], got {tuple(tokens.shape)}.")
        if tokens.shape[1] != 2 * self.pair_horizon:
            raise ValueError(
                f"Expected {2 * self.pair_horizon} RFSQ tokens, got {tokens.shape[1]}."
            )

        idx1 = tokens[:, :self.pair_horizon]
        idx2 = tokens[:, self.pair_horizon:]
        return torch.stack((idx1, idx2), dim=-1).flatten(start_dim=1)

    def _decode_interleaved_prefix(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode K complete interleaved pairs into an action chunk.

        ``tokens`` has shape [B, 2*K].  Each adjacent pair is converted to the
        corresponding RFSQ latent q1_i + q2_i.  Missing tail latent positions
        are padded and masked by the tokenizer decoder.
        """
        if tokens.ndim != 2:
            raise ValueError(f"Expected token tensor [B, T], got {tuple(tokens.shape)}.")
        if tokens.shape[1] % 2 != 0:
            raise ValueError(
                "Paired RFSQ decoding requires an even number of discrete tokens."
            )

        keep_k = tokens.shape[1] // 2
        if not 1 <= keep_k <= self.pair_horizon:
            raise ValueError(
                f"Expected 1 <= K <= {self.pair_horizon}, got K={keep_k}."
            )

        quantizer = self.action_tokenizer.quantizer
        idx1 = tokens[:, 0::2]
        idx2 = tokens[:, 1::2]

        q1 = quantizer.stage1.indices_to_embedding(idx1)
        q2_normalized = quantizer.stage2.indices_to_embedding(idx2)
        q2 = q2_normalized * quantizer.sigma2() + quantizer.mu2
        latents = q1 + q2

        if keep_k < self.pair_horizon:
            padding = torch.zeros(
                latents.shape[0],
                self.pair_horizon - keep_k,
                latents.shape[-1],
                device=latents.device,
                dtype=latents.dtype,
            )
            latents = torch.cat((latents, padding), dim=1)

        return self.action_tokenizer.decode(
            latents,
            eval_keep_k=[keep_k] * latents.shape[0],
        )

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate K complete RFSQ pairs (2*K discrete AR tokens)."""
        if use_k_tokens is None:
            keep_k = self.pair_horizon
        else:
            keep_k = min(int(use_k_tokens), self.pair_horizon)
        if keep_k < 1:
            raise ValueError(
                f"use_k_tokens must be in [1, {self.pair_horizon}], got {keep_k}."
            )

        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk

        features = self.obs_encoder(obs_dict)
        batch_size = features.shape[0]

        action_tokens = torch.full(
            (batch_size, 1),
            self.bos_id,
            dtype=torch.long,
            device=self.device,
        )
        action_tokens = self.model.generate(
            action_tokens,
            cond=features,
            max_new_tokens=2 * keep_k,
            temperature=temperature,
            top_k=topk,
        )[:, 1:]

        with torch.inference_mode():
            action_pred = self._decode_interleaved_prefix(action_tokens)

        return {
            "action": action_pred[:, :self.n_action_steps],
            "action_pred": action_pred,
        }

    def forward(self, batch) -> torch.Tensor:
        """Train on interleaved [q1_i, q2_i] next-token targets."""
        with torch.inference_mode():
            stage_major_tokens = self.action_tokenizer.tokenize(batch["action"])
            action_tokens = self._stage_major_to_interleaved(stage_major_tokens)

        batch_size = batch["action"].shape[0]
        device = batch["action"].device
        features = self.obs_encoder(batch["obs"])

        action_tokens = torch.cat(
            (
                torch.full(
                    (batch_size, 1),
                    self.bos_id,
                    dtype=torch.long,
                    device=device,
                ),
                action_tokens,
            ),
            dim=1,
        )

        logits = self.model(action_tokens[:, :-1], cond=features)
        vocab_size = logits.shape[-1]
        return F.cross_entropy(
            logits.reshape(-1, vocab_size),
            action_tokens[:, 1:].reshape(-1),
        )

