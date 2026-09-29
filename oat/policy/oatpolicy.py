import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oat.policy.base_policy import BasePolicy
from oat.policy.self_conditioned_prefix import (
    PREDICTION_GREEDY, SelfConditionedPrefixConfig)
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
# from oat.model.autoregressive.transformer import AutoregressiveModel
from oat.model.autoregressive.transformer_cache import AutoregressiveModel


class OATPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        action_tokenizer: OATTok,
        n_action_steps: int,
        n_obs_steps: int,
        # policy model params
        embed_dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        # policy inference params
        temperature: float = 1.0,
        topk: int = 10,
        # self-conditioned mixed-prefix training (training only, off by default)
        self_conditioned_prefix: Optional[Dict] = None,
    ):
        super().__init__()

        modalities = obs_encoder.modalities()
        obs_feature_dim = obs_encoder.output_feature_dim()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in shape_meta['obs'].items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr['type']
            if type in modalities:
                obs_ports.append(key)

        # freeze action tokenizer
        for param in action_tokenizer.parameters():
            param.requires_grad_(False)
        action_tokenizer.eval()

        # create AR model
        codebook_size = action_tokenizer.quantizer.codebook_size
        max_seq_len = action_tokenizer.token_seq_len
        model = AutoregressiveModel(
            vocab_size=codebook_size + 1,  # +1 for <BOS>
            max_seq_len=max_seq_len + 1,
            max_cond_len=n_obs_steps,
            cond_dim=obs_feature_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_emb=embed_dim,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
        )
        bos_id = codebook_size  # last token id for <BOS>

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.action_tokenizer = action_tokenizer
        self.model = model
        self.max_seq_len = max_seq_len
        self.bos_id = bos_id
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.temperature = temperature
        self.topk = topk
        self.codebook_size = codebook_size

        # self-conditioned mixed-prefix training config
        self.self_conditioned_prefix = SelfConditionedPrefixConfig.from_config(
            self_conditioned_prefix)
        # training progress, set by the workspace so the schedule can advance.
        # Kept as plain python ints (not buffers) to leave checkpoints unchanged.
        self._sc_epoch = 0
        self._sc_global_step = 0
        # last-forward diagnostics for logging (plain floats, no state_dict impact)
        self.last_prefix_stats: Dict[str, float] = dict()

        # report
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs_params = sum(p.numel() for p in obs_encoder.parameters() if p.requires_grad)
        obs_trainable_ratio = num_trainable_obs_params / num_obs_params
        num_tok_params = sum(p.numel() for p in action_tokenizer.parameters())
        num_trainable_tok_params = sum(p.numel() for p in action_tokenizer.parameters() if p.requires_grad)
        tok_trainable_ratio = num_trainable_tok_params / num_tok_params
        num_model_params = sum(p.numel() for p in model.parameters())
        num_trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_trainable_ratio = num_trainable_model_params / num_model_params
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc: {num_obs_params/1e6:.1f}M ({obs_trainable_ratio:.5%} trainable)\n"
            f"  act tok: {num_tok_params/1e6:.1f}M ({tok_trainable_ratio:.5%} trainable)\n"
            f"  policy : {num_model_params/1e6:.1f}M ({model_trainable_ratio:.5%} trainable)\n"
            f"  {self.self_conditioned_prefix.summary()}\n"
        )

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities
    
    def get_observation_ports(self):
        return self.obs_ports
    
    def get_policy_name(self):
        base_name = 'oatpolicy_'
        for modality in self.modalities:
            if modality != 'state':
                base_name += modality + '|'
        return base_name[:-1]

    def create_dummy_observation(self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        return super().create_dummy_observation(
            batch_size=batch_size,
            horizon=self.n_obs_steps,
            obs_key_shapes=self.obs_key_shapes,
            device=device
        )

    def set_normalizer(self, normalizer):
        self.obs_encoder.set_normalizer(normalizer)
        # self.action_tokenizer.set_normalizer(normalizer)

    def get_optimizer(
        self, 
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """Create an AdamW optimizer with weight decay for 2D parameters only."""
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.

        encoder_decay_params = []
        encoder_nodecay_params = []
        for name, param in self.obs_encoder.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                encoder_decay_params.append(param)
            else:
                encoder_nodecay_params.append(param)

        policy_decay_params = []
        policy_nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                policy_decay_params.append(param)
            else:
                policy_nodecay_params.append(param)
        
        optim_groups = [
            {'params': policy_decay_params, 'lr': policy_lr, 'weight_decay': weight_decay},
            {'params': policy_nodecay_params, 'lr': policy_lr, 'weight_decay': 0.0},
            {'params': encoder_decay_params, 'lr': obs_enc_lr, 'weight_decay': weight_decay},
            {'params': encoder_nodecay_params, 'lr': obs_enc_lr, 'weight_decay': 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, betas=betas)
        return optimizer

    def predict_action(self, 
        obs_dict: Dict[str, torch.Tensor],
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if use_k_tokens is None:
            use_k_tokens = self.max_seq_len
        else:
            use_k_tokens = min(use_k_tokens, self.max_seq_len)
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk

        # encode observation
        features = self.obs_encoder(obs_dict)   # [B, To, d]
        B = features.shape[0]

        # autoregressive generation
        action_tokens = torch.full( # [B, 1] seq: [<BOS>,]
            (B, 1), self.bos_id, 
            dtype=torch.long, device=self.device
        )
        action_tokens = self.model.generate(
            action_tokens,
            cond=features,
            max_new_tokens=use_k_tokens,
            temperature=temperature,
            top_k=topk,
        )[:, 1:]    # [B, max_seq_len], drop <BOS>

        # decode action tokens
        with torch.inference_mode():
            action_pred = self.action_tokenizer.detokenize(
                tokens=action_tokens,
            )

        # receeding horizon
        action = action_pred[:,:self.n_action_steps]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result


    def set_training_progress(self, epoch: int = 0, global_step: int = 0):
        """Let the workspace advance the self-prefix probability schedule.

        No-op for the default (disabled / constant) configuration, so existing
        workspaces that never call it stay correct.
        """
        self._sc_epoch = int(epoch)
        self._sc_global_step = int(global_step)

    def get_prefix_stats(self) -> Dict[str, float]:
        """Diagnostics from the most recent self-conditioned forward pass."""
        return dict(self.last_prefix_stats)

    def forward(self, batch) -> torch.Tensor:
        if self.self_conditioned_prefix.enabled and self.training:
            return self._forward_self_conditioned(batch)
        return self._forward_teacher_forcing(batch)

    def _tokenize_with_bos(self, batch) -> torch.LongTensor:
        """GT token sequence with <BOS> prepended: [B, 1 + max_seq_len]."""
        # tokenize trajectory
        with torch.inference_mode():
            action_tokens = self.action_tokenizer.tokenize(batch['action'])

        B = batch['action'].shape[0]
        device = batch['action'].device

        # prepend <BOS> token
        action_tokens = torch.cat([
            torch.full(
                (B, 1), self.bos_id,
                dtype=torch.long, device=device
            ),
            action_tokens
        ], dim=1)
        return action_tokens

    def _forward_teacher_forcing(self, batch) -> torch.Tensor:
        """Original OAT training path: a single parallel teacher-forced pass."""
        action_tokens = self._tokenize_with_bos(batch)

        # encode observation
        features = self.obs_encoder(batch['obs'])   # [B, To, d]

        # forward model
        logits = self.model(action_tokens[:, :-1], cond=features)

        # compute loss
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),     # (B*T, vocab_size)
            action_tokens[:, 1:].reshape(-1)    # (B*T,)
        )
        return loss

    def _forward_self_conditioned(self, batch) -> torch.Tensor:
        """Self-conditioned mixed-prefix training.

        The prefix fed to the model is a per-sample / per-register random mix of
        GT tokens (anchors) and the policy's own greedy predictions.  The
        cross-entropy target at every register is always the clean GT token, so
        the model learns ``imperfect history -> correct next token``.

        Inference is untouched: this path only runs in ``self.training`` mode
        when ``self_conditioned_prefix.enabled`` is set.
        """
        cfg = self.self_conditioned_prefix
        action_tokens = self._tokenize_with_bos(batch)      # [B, 1 + T]
        gt_tokens = action_tokens[:, 1:]                    # [B, T] clean targets
        B, T = gt_tokens.shape
        device = gt_tokens.device

        # encode observation (same encoder call as teacher forcing)
        features = self.obs_encoder(batch['obs'])   # [B, To, d]

        # cross-attention memory is static across decoding steps
        memory, memory_kv_cache = self.model.encode_cond(features)

        self_prob = cfg.current_self_prob(
            epoch=self._sc_epoch, global_step=self._sc_global_step)

        # step 0 input is always <BOS>
        step_tokens = action_tokens[:, :1]                  # [B, 1]
        past_key_values = None
        losses = []
        self_flags = []

        for k in range(T):
            logits, past_key_values = self.model.forward_step(
                step_tokens,
                memory=memory,
                past_key_values=past_key_values,
                memory_kv_cache=memory_kv_cache,
                pos_offset=k,
            )
            step_logits = logits[:, -1, :]                  # [B, vocab_size]

            # supervision is always the clean GT token for register k
            losses.append(F.cross_entropy(
                step_logits, gt_tokens[:, k], reduction='none'))    # [B]

            if k == T - 1:
                # no further prefix slot to fill
                break

            # --- build the input token for register k (used at step k+1) ---
            gt_k = gt_tokens[:, k]                                  # [B]
            if self_prob > 0.0:
                if cfg.prediction_mode != PREDICTION_GREEDY:
                    raise NotImplementedError(
                        f"prediction_mode={cfg.prediction_mode!r} is not implemented."
                    )
                # greedy argmax restricted to the action vocabulary so that
                # <BOS> can never be fed back as an action token
                pred_k = step_logits[:, :self.codebook_size].argmax(dim=-1).detach()
                assert bool((pred_k >= 0).all() and (pred_k < self.codebook_size).all()), \
                    "self-predicted prefix token outside the action vocabulary"

                use_self = torch.rand(B, device=device) < self_prob  # per-sample
                next_token = torch.where(use_self, pred_k, gt_k)
            else:
                use_self = torch.zeros(B, dtype=torch.bool, device=device)
                next_token = gt_k

            self_flags.append(use_self)
            step_tokens = next_token.detach().unsqueeze(1)           # [B, 1]

        per_register_loss = torch.stack(losses, dim=1)               # [B, T]
        loss = per_register_loss.mean()

        # --- diagnostics (detached floats, safe to log) ---
        with torch.no_grad():
            stats = {
                'self_prefix_prob': float(self_prob),
                'loss_total': float(loss.detach()),
            }
            reg_loss = per_register_loss.detach().mean(dim=0)
            for k in range(T):
                stats[f'loss_register_{k}'] = float(reg_loss[k])
            if self_flags:
                flags = torch.stack(self_flags, dim=1).float()       # [B, T-1]
                stats['self_prefix_fraction'] = float(flags.mean())
                stats['gt_anchor_fraction'] = float(1.0 - flags.mean())
                frac_per_slot = flags.mean(dim=0)
                for k in range(frac_per_slot.shape[0]):
                    stats[f'self_prefix_fraction_slot_{k}'] = float(frac_per_slot[k])
            else:
                stats['self_prefix_fraction'] = 0.0
                stats['gt_anchor_fraction'] = 1.0
            self.last_prefix_stats = stats

        return loss
