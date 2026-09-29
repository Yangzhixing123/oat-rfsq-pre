"""Sanity checks for self-conditioned mixed-prefix OAT policy training.

Run with:
    uv run python tests/test_self_conditioned_prefix.py

These are small synthetic-tensor tests.  They build a real ``AutoregressiveModel``
but stub out the observation encoder and action tokenizer, so no dataset,
checkpoint, or simulator is needed.
"""

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
from oat.policy.oatpolicy import OATPolicy
from oat.policy.self_conditioned_prefix import SelfConditionedPrefixConfig


CODEBOOK_SIZE = 1000
NUM_REGISTERS = 8
COND_LEN = 2
COND_DIM = 16
EMB_DIM = 32


class _StubEncoder(nn.Module):
    """Deterministic-but-learnable stand-in for the observation encoder."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(COND_DIM, COND_DIM)

    def forward(self, obs_dict):
        return self.proj(obs_dict['feat'])


class _StubTokenizer:
    """Returns a fixed GT token sequence, mimicking OATTok.tokenize."""

    def __init__(self, tokens: torch.LongTensor):
        self.tokens = tokens

    def tokenize(self, action):
        return self.tokens[:action.shape[0]].clone()


def build_policy(self_conditioned_prefix=None, seed: int = 0) -> OATPolicy:
    """Construct an OATPolicy-like object without the real encoder/tokenizer."""
    torch.manual_seed(seed)

    policy = OATPolicy.__new__(OATPolicy)
    nn.Module.__init__(policy)
    policy._dummy_variable = nn.Parameter(torch.zeros(1), requires_grad=False)

    policy.model = AutoregressiveModel(
        vocab_size=CODEBOOK_SIZE + 1,
        max_seq_len=NUM_REGISTERS + 1,
        max_cond_len=COND_LEN,
        cond_dim=COND_DIM,
        n_layer=2,
        n_head=2,
        n_emb=EMB_DIM,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
    )
    policy.obs_encoder = _StubEncoder()
    policy.max_seq_len = NUM_REGISTERS
    policy.codebook_size = CODEBOOK_SIZE
    policy.bos_id = CODEBOOK_SIZE
    policy.self_conditioned_prefix = SelfConditionedPrefixConfig.from_config(
        self_conditioned_prefix)
    policy._sc_epoch = 0
    policy._sc_global_step = 0
    policy.last_prefix_stats = dict()
    return policy


def make_batch(batch_size: int = 4, seed: int = 123):
    gen = torch.Generator().manual_seed(seed)
    gt_tokens = torch.randint(
        0, CODEBOOK_SIZE, (batch_size, NUM_REGISTERS), generator=gen)
    batch = {
        'action': torch.randn(batch_size, 32, 7, generator=gen),
        'obs': {'feat': torch.randn(batch_size, COND_LEN, COND_DIM, generator=gen)},
    }
    return batch, gt_tokens


def _attach_tokenizer(policy, gt_tokens):
    policy.action_tokenizer = _StubTokenizer(gt_tokens)


# ---------------------------------------------------------------- tests


def test_disabled_matches_original():
    """1. enabled=false -> exactly the original teacher-forcing loss."""
    batch, gt = make_batch()
    policy = build_policy({'enabled': False})
    _attach_tokenizer(policy, gt)
    policy.eval()

    with torch.no_grad():
        loss_new = policy(batch)

        # replicate the original forward() verbatim
        bos = torch.full((gt.shape[0], 1), policy.bos_id, dtype=torch.long)
        tokens = torch.cat([bos, gt], dim=1)
        features = policy.obs_encoder(batch['obs'])
        logits = policy.model(tokens[:, :-1], cond=features)
        loss_ref = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))

    assert torch.allclose(loss_new, loss_ref, atol=0, rtol=0), (loss_new, loss_ref)
    print(f"[ok] 1. enabled=false loss == original: {loss_new.item():.6f}")


def test_disabled_train_mode_uses_parallel_path():
    """1b. enabled=false in train() mode still takes the parallel path."""
    batch, gt = make_batch()
    policy = build_policy({'enabled': False})
    _attach_tokenizer(policy, gt)
    policy.train()

    calls = {'step': 0, 'parallel': 0}
    orig_step = policy.model.forward_step
    orig_forward = policy.model.forward

    def counting_step(*a, **kw):
        calls['step'] += 1
        return orig_step(*a, **kw)

    def counting_forward(*a, **kw):
        calls['parallel'] += 1
        return orig_forward(*a, **kw)

    policy.model.forward_step = counting_step
    policy.model.forward = counting_forward
    policy(batch)
    policy.model.forward_step = orig_step
    policy.model.forward = orig_forward

    assert calls['step'] == 0, calls
    assert calls['parallel'] == 1, calls
    print("[ok] 1b. enabled=false uses 1 parallel forward, 0 sequential steps")


def test_self_prob_zero_matches_teacher_forcing():
    """2. self_prob=0 -> sequential path is numerically teacher forcing."""
    batch, gt = make_batch()
    policy = build_policy({'enabled': True, 'self_prob': 0.0})
    _attach_tokenizer(policy, gt)
    policy.train()  # sequential path is train-only
    policy.model.eval()      # disable dropout so the two paths are comparable
    policy.obs_encoder.eval()

    with torch.no_grad():
        loss_seq = policy(batch)
        loss_tf = policy._forward_teacher_forcing(batch)

    assert torch.allclose(loss_seq, loss_tf, atol=1e-5), (loss_seq, loss_tf)
    stats = policy.get_prefix_stats()
    assert stats['self_prefix_fraction'] == 0.0
    assert stats['gt_anchor_fraction'] == 1.0
    print(
        f"[ok] 2. self_prob=0 sequential loss {loss_seq.item():.6f} "
        f"== teacher forcing {loss_tf.item():.6f}"
    )


def test_self_prob_one_uses_only_predictions():
    """3. self_prob=1 -> every prefix slot after <BOS> is self-predicted."""
    batch, gt = make_batch()
    policy = build_policy({'enabled': True, 'self_prob': 1.0})
    _attach_tokenizer(policy, gt)
    policy.train()
    policy.model.eval()
    policy.obs_encoder.eval()

    fed_tokens = []
    orig_step = policy.model.forward_step

    def recording_step(tokens, *a, **kw):
        fed_tokens.append(tokens.clone())
        return orig_step(tokens, *a, **kw)

    policy.model.forward_step = recording_step
    with torch.no_grad():
        policy(batch)
    policy.model.forward_step = orig_step

    assert len(fed_tokens) == NUM_REGISTERS, len(fed_tokens)
    # step 0 is <BOS>
    assert bool((fed_tokens[0] == policy.bos_id).all())
    # steps 1..T-1 must differ from GT wherever the model disagrees, and must
    # never equal <BOS>
    n_matching_gt = 0
    for k in range(1, NUM_REGISTERS):
        fed = fed_tokens[k].squeeze(1)
        assert bool((fed < CODEBOOK_SIZE).all()), "fed <BOS> or out-of-vocab token"
        assert bool((fed >= 0).all())
        n_matching_gt += int((fed == gt[:, k - 1]).sum())
    stats = policy.get_prefix_stats()
    assert stats['self_prefix_fraction'] == 1.0
    assert stats['gt_anchor_fraction'] == 0.0
    print(
        "[ok] 3. self_prob=1 feeds only self-predictions "
        f"({n_matching_gt} of {(NUM_REGISTERS-1)*gt.shape[0]} coincidentally == GT)"
    )


def test_targets_invariant_to_mask():
    """4. GT targets never change, whatever the prefix mask is."""
    batch, gt = make_batch()
    seen_targets = []

    orig_ce = F.cross_entropy

    for self_prob in (0.0, 0.25, 0.5, 1.0):
        policy = build_policy({'enabled': True, 'self_prob': self_prob})
        _attach_tokenizer(policy, gt)
        policy.train()
        policy.model.eval()
        policy.obs_encoder.eval()

        targets = []

        def recording_ce(logits, target, **kw):
            targets.append(target.clone())
            return orig_ce(logits, target, **kw)

        F.cross_entropy = recording_ce
        try:
            with torch.no_grad():
                policy(batch)
        finally:
            F.cross_entropy = orig_ce

        assert len(targets) == NUM_REGISTERS
        stacked = torch.stack(targets, dim=1)
        assert torch.equal(stacked, gt), "CE targets deviated from clean GT"
        seen_targets.append(stacked)

    for other in seen_targets[1:]:
        assert torch.equal(seen_targets[0], other)
    print("[ok] 4. CE targets == clean GT for self_prob in {0, .25, .5, 1}")


def test_detach_no_cross_step_gradient():
    """9. self-predicted prefix tokens carry no gradient across steps."""
    batch, gt = make_batch()
    policy = build_policy({'enabled': True, 'self_prob': 1.0})
    _attach_tokenizer(policy, gt)
    policy.train()
    policy.model.eval()
    policy.obs_encoder.eval()

    fed = []
    orig_step = policy.model.forward_step

    def recording_step(tokens, *a, **kw):
        fed.append(tokens)
        return orig_step(tokens, *a, **kw)

    policy.model.forward_step = recording_step
    loss = policy(batch)
    policy.model.forward_step = orig_step

    for tokens in fed:
        assert tokens.dtype == torch.long, tokens.dtype
        assert not tokens.requires_grad
        assert tokens.grad_fn is None

    loss.backward()
    n_with_grad = sum(
        1 for p in policy.model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_with_grad > 0, "no gradient reached the AR model"
    print(
        "[ok] 9. fed prefix tokens are detached LongTensors; "
        f"{n_with_grad} model tensors received gradient"
    )


def test_shapes_and_per_register_supervision():
    """7 + 8. all 8 registers get one clean-GT CE term."""
    batch, gt = make_batch()
    policy = build_policy({'enabled': True, 'self_prob': 0.5})
    _attach_tokenizer(policy, gt)
    policy.train()
    policy.model.eval()
    policy.obs_encoder.eval()

    with torch.no_grad():
        policy(batch)
    stats = policy.get_prefix_stats()
    per_reg = [k for k in stats if k.startswith('loss_register_')]
    assert len(per_reg) == NUM_REGISTERS, per_reg
    assert all(stats[k] > 0 for k in per_reg)
    slot_keys = [k for k in stats if k.startswith('self_prefix_fraction_slot_')]
    assert len(slot_keys) == NUM_REGISTERS - 1, slot_keys
    print(f"[ok] 7/8. {NUM_REGISTERS} per-register losses logged: "
          + ", ".join(f"{stats[f'loss_register_{k}']:.3f}" for k in range(NUM_REGISTERS)))


def test_vocabulary_bounds_under_bos_bias():
    """5 + 6. even if <BOS> dominates the logits it is never fed back."""
    batch, gt = make_batch()
    policy = build_policy({'enabled': True, 'self_prob': 1.0})
    _attach_tokenizer(policy, gt)
    policy.train()
    policy.model.eval()
    policy.obs_encoder.eval()

    # make <BOS> the global argmax by biasing its output embedding row
    with torch.no_grad():
        policy.model.tok_emb.weight[policy.bos_id] *= 1e3

    fed = []
    orig_step = policy.model.forward_step

    def recording_step(tokens, *a, **kw):
        fed.append(tokens.clone())
        logits, kv = orig_step(tokens, *a, **kw)
        return logits, kv

    policy.model.forward_step = recording_step
    with torch.no_grad():
        policy(batch)
    policy.model.forward_step = orig_step

    # confirm the unrestricted argmax really would have been <BOS>
    with torch.no_grad():
        features = policy.obs_encoder(batch['obs'])
        bos = torch.full((gt.shape[0], 1), policy.bos_id, dtype=torch.long)
        logits0 = policy.model(bos, cond=features)[:, -1, :]
        unrestricted = logits0.argmax(dim=-1)
    assert bool((unrestricted == policy.bos_id).all()), \
        "test setup failed: <BOS> is not the argmax"

    for tokens in fed[1:]:
        assert bool((tokens < CODEBOOK_SIZE).all()), "fed <BOS> back as an action token"
        assert bool((tokens >= 0).all())
    print("[ok] 5/6. <BOS> is argmax yet never enters the prefix "
          f"(fed max id {int(torch.cat(fed[1:]).max())} < {CODEBOOK_SIZE})")


def test_deterministic_mask_under_fixed_seed():
    """10. fixed seed -> reproducible mask and loss."""
    batch, gt = make_batch()

    def run():
        policy = build_policy({'enabled': True, 'self_prob': 0.5})
        _attach_tokenizer(policy, gt)
        policy.train()
        policy.model.eval()
        policy.obs_encoder.eval()
        torch.manual_seed(7)
        with torch.no_grad():
            loss = policy(batch)
        return loss.item(), policy.get_prefix_stats()

    loss_a, stats_a = run()
    loss_b, stats_b = run()
    assert loss_a == loss_b, (loss_a, loss_b)
    for key in stats_a:
        assert stats_a[key] == stats_b[key], (key, stats_a[key], stats_b[key])
    print(f"[ok] 10. seed 7 reproducible: loss={loss_a:.6f}, "
          f"self_prefix_fraction={stats_a['self_prefix_fraction']:.4f}")


def test_forward_step_matches_parallel_forward():
    """KV-cached step loop == parallel forward on the same token sequence."""
    batch, gt = make_batch()
    policy = build_policy({'enabled': True, 'self_prob': 0.0})
    _attach_tokenizer(policy, gt)
    policy.eval()

    bos = torch.full((gt.shape[0], 1), policy.bos_id, dtype=torch.long)
    tokens = torch.cat([bos, gt], dim=1)[:, :-1]     # [B, T]

    with torch.no_grad():
        features = policy.obs_encoder(batch['obs'])
        parallel = policy.model(tokens, cond=features)

        memory, memory_kv = policy.model.encode_cond(features)
        past = None
        step_logits = []
        for k in range(tokens.shape[1]):
            out, past = policy.model.forward_step(
                tokens[:, k:k + 1], memory=memory,
                past_key_values=past, memory_kv_cache=memory_kv, pos_offset=k)
            step_logits.append(out[:, -1, :])
        stepwise = torch.stack(step_logits, dim=1)

    max_diff = (parallel - stepwise).abs().max().item()
    assert max_diff < 1e-4, max_diff
    print(f"[ok] extra. forward_step loop == model.forward (max |diff| = {max_diff:.2e})")


def test_schedule():
    """Linear schedule ramps from start_prob to end_prob."""
    cfg = SelfConditionedPrefixConfig.from_config({
        'enabled': True, 'schedule': 'linear',
        'start_prob': 0.0, 'end_prob': 0.5, 'warmup_epochs': 100,
    })
    assert cfg.current_self_prob(epoch=0) == 0.0
    assert abs(cfg.current_self_prob(epoch=50) - 0.25) < 1e-9
    assert abs(cfg.current_self_prob(epoch=100) - 0.5) < 1e-9
    assert abs(cfg.current_self_prob(epoch=400) - 0.5) < 1e-9

    cfg_steps = SelfConditionedPrefixConfig.from_config({
        'enabled': True, 'schedule': 'linear',
        'start_prob': 0.1, 'end_prob': 0.5,
        'warmup_epochs': None, 'warmup_steps': 1000,
    })
    assert abs(cfg_steps.current_self_prob(global_step=500) - 0.3) < 1e-9

    off = SelfConditionedPrefixConfig.from_config({'enabled': False, 'self_prob': 0.9})
    assert off.current_self_prob(epoch=999) == 0.0

    for bad in ({'schedule': 'cosine'}, {'prediction_mode': 'sample'},
                {'self_prob': 1.5}, {'not_a_key': 1}):
        try:
            SelfConditionedPrefixConfig.from_config(bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"expected rejection of {bad}")
    print("[ok] extra. linear/step schedules and config validation behave")


def test_schedule_end_to_end():
    """set_training_progress actually changes the realised self-prefix fraction."""
    batch, gt = make_batch(batch_size=64)
    policy = build_policy({
        'enabled': True, 'schedule': 'linear',
        'start_prob': 0.0, 'end_prob': 1.0, 'warmup_epochs': 100,
    })
    _attach_tokenizer(policy, gt)
    policy.train()
    policy.model.eval()
    policy.obs_encoder.eval()

    fractions = []
    for epoch in (0, 50, 100):
        policy.set_training_progress(epoch=epoch, global_step=epoch * 10)
        with torch.no_grad():
            policy(batch)
        stats = policy.get_prefix_stats()
        fractions.append((epoch, stats['self_prefix_prob'], stats['self_prefix_fraction']))

    assert fractions[0][1] == 0.0 and fractions[0][2] == 0.0
    assert abs(fractions[1][1] - 0.5) < 1e-9
    assert 0.3 < fractions[1][2] < 0.7
    assert fractions[2][1] == 1.0 and fractions[2][2] == 1.0
    print("[ok] extra. schedule end-to-end: "
          + ", ".join(f"ep{e}: p={p:.2f} frac={f:.3f}" for e, p, f in fractions))


TESTS = [
    test_disabled_matches_original,
    test_disabled_train_mode_uses_parallel_path,
    test_self_prob_zero_matches_teacher_forcing,
    test_self_prob_one_uses_only_predictions,
    test_targets_invariant_to_mask,
    test_vocabulary_bounds_under_bos_bias,
    test_shapes_and_per_register_supervision,
    test_detach_no_cross_step_gradient,
    test_deterministic_mask_under_fixed_seed,
    test_forward_step_matches_parallel_forward,
    test_schedule,
    test_schedule_end_to_end,
]


if __name__ == '__main__':
    failures = 0
    for test in TESTS:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {test.__name__}: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} checks passed")
    sys.exit(1 if failures else 0)
