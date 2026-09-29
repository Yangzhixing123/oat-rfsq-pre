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
from oat.policy.self_conditioned_prefix import PREDICTION_GREEDY


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

    def _tokenize_with_bos(self, batch) -> torch.LongTensor:
        """Return clean interleaved RFSQ targets with BOS prepended.

        Overriding this hook makes both the inherited parallel teacher-forcing
        path and the pair-aware self-conditioned path use
        q1_1, q2_1, ..., q1_H, q2_H ordering.
        """
        with torch.inference_mode():
            stage_major_tokens = self.action_tokenizer.tokenize(batch["action"])
            action_tokens = self._stage_major_to_interleaved(stage_major_tokens)

        batch_size = batch["action"].shape[0]
        device = batch["action"].device

        return torch.cat(
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

    def _forward_self_conditioned(self, batch) -> torch.Tensor:
        """Train with one shared self/GT decision per complete RFSQ pair.

        For pair i, one Bernoulli decision controls both prefix tokens q1_i
        and q2_i. A self-conditioned pair feeds detached greedy predictions
        for both tokens; a GT-anchor pair feeds both clean tokens.
        Supervision always remains the clean interleaved target sequence.
        """
        cfg = self.self_conditioned_prefix
        action_tokens = self._tokenize_with_bos(batch)  # [B, 1 + 2*H]
        gt_tokens = action_tokens[:, 1:]                # [B, 2*H]
        batch_size, token_count = gt_tokens.shape
        device = gt_tokens.device

        if token_count != 2 * self.pair_horizon:
            raise ValueError(
                f"Expected {2 * self.pair_horizon} interleaved RFSQ targets, "
                f"got {token_count}."
            )

        features = self.obs_encoder(batch["obs"])
        memory, memory_kv_cache = self.model.encode_cond(features)
        self_prob = cfg.current_self_prob(
            epoch=self._sc_epoch,
            global_step=self._sc_global_step,
        )

        step_tokens = action_tokens[:, :1]  # first input is always BOS
        past_key_values = None
        losses = []
        prefix_flags = []
        pair_flags = []
        current_pair_uses_self = None

        for token_idx in range(token_count):
            logits, past_key_values = self.model.forward_step(
                step_tokens,
                memory=memory,
                past_key_values=past_key_values,
                memory_kv_cache=memory_kv_cache,
                pos_offset=token_idx,
            )
            step_logits = logits[:, -1, :]
            losses.append(F.cross_entropy(
                step_logits,
                gt_tokens[:, token_idx],
                reduction="none",
            ))

            # q1_i starts a pair. Draw once and reuse the same decision for
            # q2_i, so a pair is never half self-conditioned and half GT.
            if token_idx % 2 == 0:
                if self_prob > 0.0:
                    current_pair_uses_self = (
                        torch.rand(batch_size, device=device) < self_prob
                    )
                else:
                    current_pair_uses_self = torch.zeros(
                        batch_size,
                        dtype=torch.bool,
                        device=device,
                    )
                pair_flags.append(current_pair_uses_self)

            if token_idx == token_count - 1:
                break

            if self_prob > 0.0:
                if cfg.prediction_mode != PREDICTION_GREEDY:
                    raise NotImplementedError(
                        f"prediction_mode={cfg.prediction_mode!r} is not implemented."
                    )
                pred_token = step_logits[:, :self.codebook_size].argmax(
                    dim=-1
                ).detach()
                next_token = torch.where(
                    current_pair_uses_self,
                    pred_token,
                    gt_tokens[:, token_idx],
                )
            else:
                next_token = gt_tokens[:, token_idx]

            prefix_flags.append(current_pair_uses_self)
            step_tokens = next_token.detach().unsqueeze(1)

        per_token_loss = torch.stack(losses, dim=1)  # [B, 2*H]
        loss = per_token_loss.mean()

        with torch.no_grad():
            stats = {
                "self_prefix_prob": float(self_prob),
                "loss_total": float(loss.detach()),
            }

            token_loss = per_token_loss.detach().mean(dim=0)
            for token_idx in range(token_count):
                stats[f"loss_register_{token_idx}"] = float(token_loss[token_idx])
            pair_loss = per_token_loss.detach().reshape(
                batch_size, self.pair_horizon, 2
            ).mean(dim=(0, 2))
            for pair_idx in range(self.pair_horizon):
                stats[f"loss_pair_{pair_idx}"] = float(pair_loss[pair_idx])

            prefix_mask = torch.stack(prefix_flags, dim=1).float()
            pair_mask = torch.stack(pair_flags, dim=1).float()
            stats["self_prefix_fraction"] = float(prefix_mask.mean())
            stats["gt_anchor_fraction"] = float(1.0 - prefix_mask.mean())
            stats["self_pair_fraction"] = float(pair_mask.mean())
            stats["gt_pair_fraction"] = float(1.0 - pair_mask.mean())

            prefix_fraction_per_slot = prefix_mask.mean(dim=0)
            for token_idx in range(prefix_fraction_per_slot.shape[0]):
                stats[f"self_prefix_fraction_slot_{token_idx}"] = float(
                    prefix_fraction_per_slot[token_idx]
                )
            pair_fraction_per_slot = pair_mask.mean(dim=0)
            for pair_idx in range(pair_fraction_per_slot.shape[0]):
                stats[f"self_pair_fraction_pair_{pair_idx}"] = float(
                    pair_fraction_per_slot[pair_idx]
                )
            self.last_prefix_stats = stats

        return loss

