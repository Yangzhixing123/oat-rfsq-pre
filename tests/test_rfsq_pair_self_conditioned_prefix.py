"""Tests for pair-aware self-conditioned RFSQ policy training."""

from __future__ import annotations

import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from oat.model.autoregressive.transformer_cache import AutoregressiveModel
from oat.policy.oatpolicy_rfsq_pair import RFSQPairOATPolicy
from oat.policy.self_conditioned_prefix import SelfConditionedPrefixConfig


CODEBOOK_SIZE = 17
PAIR_HORIZON = 4
TOKEN_COUNT = 2 * PAIR_HORIZON
COND_LEN = 2
COND_DIM = 8
EMB_DIM = 16


class _StubEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(COND_DIM, COND_DIM)

    def forward(self, obs_dict):
        return self.proj(obs_dict["feat"])


class _StubTokenizer:
    def __init__(self, stage_major_tokens: torch.LongTensor):
        self.stage_major_tokens = stage_major_tokens

    def tokenize(self, action):
        return self.stage_major_tokens[:action.shape[0]].clone()


def _make_batch_and_stage_major(batch_size: int = 4, seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    stage1 = torch.randint(
        0, CODEBOOK_SIZE, (batch_size, PAIR_HORIZON), generator=generator
    )
    stage2 = torch.randint(
        0, CODEBOOK_SIZE, (batch_size, PAIR_HORIZON), generator=generator
    )
    stage_major = torch.cat((stage1, stage2), dim=1)
    interleaved = torch.stack((stage1, stage2), dim=-1).flatten(start_dim=1)
    batch = {
        "action": torch.randn(batch_size, 32, 7, generator=generator),
        "obs": {
            "feat": torch.randn(
                batch_size, COND_LEN, COND_DIM, generator=generator
            )
        },
    }
    return batch, stage_major, interleaved


def _build_policy(stage_major, config, seed: int = 0):
    torch.manual_seed(seed)
    policy = RFSQPairOATPolicy.__new__(RFSQPairOATPolicy)
    nn.Module.__init__(policy)
    policy._dummy_variable = nn.Parameter(torch.zeros(1), requires_grad=False)
    policy.model = AutoregressiveModel(
        vocab_size=CODEBOOK_SIZE + 1,
        max_seq_len=TOKEN_COUNT + 1,
        max_cond_len=COND_LEN,
        cond_dim=COND_DIM,
        n_layer=2,
        n_head=2,
        n_emb=EMB_DIM,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    )
    policy.obs_encoder = _StubEncoder()
    policy.action_tokenizer = _StubTokenizer(stage_major)
    policy.max_seq_len = TOKEN_COUNT
    policy.pair_horizon = PAIR_HORIZON
    policy.codebook_size = CODEBOOK_SIZE
    policy.bos_id = CODEBOOK_SIZE
    policy.self_conditioned_prefix = SelfConditionedPrefixConfig.from_config(config)
    policy._sc_epoch = 0
    policy._sc_global_step = 0
    policy.last_prefix_stats = {}
    return policy


def test_disabled_path_is_parallel_and_interleaved():
    batch, stage_major, interleaved = _make_batch_and_stage_major()
    policy = _build_policy(stage_major, {"enabled": False})
    policy.train()

    calls = {"parallel": 0, "step": 0}
    seen_parallel_input = []
    original_forward = policy.model.forward
    original_step = policy.model.forward_step

    def record_forward(tokens, *args, **kwargs):
        calls["parallel"] += 1
        seen_parallel_input.append(tokens.detach().clone())
        return original_forward(tokens, *args, **kwargs)

    def record_step(*args, **kwargs):
        calls["step"] += 1
        return original_step(*args, **kwargs)

    policy.model.forward = record_forward
    policy.model.forward_step = record_step
    policy(batch)

    expected = torch.cat(
        (
            torch.full((interleaved.shape[0], 1), policy.bos_id),
            interleaved[:, :-1],
        ),
        dim=1,
    )
    assert calls == {"parallel": 1, "step": 0}
    assert torch.equal(seen_parallel_input[0], expected)


def test_self_prob_zero_matches_interleaved_teacher_forcing():
    batch, stage_major, _ = _make_batch_and_stage_major()
    policy = _build_policy(stage_major, {"enabled": True, "self_prob": 0.0})
    policy.train()
    policy.model.eval()
    policy.obs_encoder.eval()

    with torch.no_grad():
        sequential_loss = policy(batch)
        teacher_forcing_loss = policy._forward_teacher_forcing(batch)

    assert torch.allclose(sequential_loss, teacher_forcing_loss, atol=1e-5)
    stats = policy.get_prefix_stats()
    assert stats["self_pair_fraction"] == 0.0
    assert stats["gt_pair_fraction"] == 1.0


def test_pair_mask_is_shared_and_targets_stay_clean(monkeypatch):
    batch, stage_major, interleaved = _make_batch_and_stage_major()
    policy = _build_policy(stage_major, {"enabled": True, "self_prob": 0.25})
    policy.train()
    policy.model.eval()
    policy.obs_encoder.eval()

    draws = [
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
    ]
    expected_pair_masks = [draw < 0.25 for draw in draws]
    remaining_draws = [draw.clone() for draw in draws]

    def fixed_rand(*size, **kwargs):
        assert size == (interleaved.shape[0],)
        value = remaining_draws.pop(0)
        return value.to(device=kwargs.get("device"))

    monkeypatch.setattr(torch, "rand", fixed_rand)

    fed_tokens = []
    greedy_predictions = []
    clean_targets = []
    original_step = policy.model.forward_step
    original_cross_entropy = F.cross_entropy

    def record_step(tokens, *args, **kwargs):
        fed_tokens.append(tokens.detach().clone())
        logits, cache = original_step(tokens, *args, **kwargs)
        greedy_predictions.append(
            logits[:, -1, :CODEBOOK_SIZE].argmax(dim=-1).detach()
        )
        return logits, cache

    def record_cross_entropy(logits, targets, **kwargs):
        clean_targets.append(targets.detach().clone())
        return original_cross_entropy(logits, targets, **kwargs)

    monkeypatch.setattr(policy.model, "forward_step", record_step)
    monkeypatch.setattr(F, "cross_entropy", record_cross_entropy)

    with torch.no_grad():
        policy(batch)

    assert not remaining_draws
    assert torch.equal(torch.stack(clean_targets, dim=1), interleaved)

    for pair_idx, pair_mask in enumerate(expected_pair_masks):
        q1_idx = 2 * pair_idx
        q2_idx = q1_idx + 1

        expected_q1_prefix = torch.where(
            pair_mask,
            greedy_predictions[q1_idx],
            interleaved[:, q1_idx],
        )
        assert torch.equal(
            fed_tokens[q1_idx + 1].squeeze(1),
            expected_q1_prefix,
        )

        if q2_idx + 1 < TOKEN_COUNT:
            expected_q2_prefix = torch.where(
                pair_mask,
                greedy_predictions[q2_idx],
                interleaved[:, q2_idx],
            )
            assert torch.equal(
                fed_tokens[q2_idx + 1].squeeze(1),
                expected_q2_prefix,
            )

    stats = policy.get_prefix_stats()
    expected_fraction = torch.stack(expected_pair_masks).float().mean().item()
    assert abs(stats["self_pair_fraction"] - expected_fraction) < 1e-7
    assert abs(stats["gt_pair_fraction"] - (1.0 - expected_fraction)) < 1e-7
