import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from oat.model.common.module_attr_mixin import ModuleAttrMixin


class MLP(nn.Module):
    """ Standard Feed-Forward Network for a Transformer Block """
    def __init__(self, n_emb: int, p_drop: float):
        super().__init__()
        self.c_fc   = nn.Linear(n_emb, 4 * n_emb)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * n_emb, n_emb)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class CausalSelfAttention(nn.Module):
    """ Causal Self-Attention with KV Caching """
    def __init__(self, n_head: int, n_emb: int, p_drop_attn: float):
        super().__init__()
        assert n_emb % n_head == 0
        # Q, K, V projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_emb, 3 * n_emb, bias=False)
        self.c_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.resid_dropout = nn.Dropout(p_drop_attn)
        self.p_attn_dropout = p_drop_attn
        
        self.n_head = n_head
        self.n_emb = n_emb
        self.head_dim = n_emb // n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_emb, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.n_head, self.head_dim).transpose(1, 2), (q, k, v))

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        
        # Use causal masking only when:
        # 1. No past cache (training or first forward in generation)
        # 2. Query sequence length > 1
        # When we have past cache, we're generating one token at a time,
        # and the token can attend to all previous tokens (which are in the cache)
        is_causal = (layer_past is None) and (T > 1)
        
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            is_causal=is_causal,
            dropout_p=self.p_attn_dropout if self.training else 0.0
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, (k, v)


class CrossAttention(nn.Module):
    """ Cross-Attention module """
    def __init__(self, n_head: int, n_emb: int, p_drop_attn: float):
        super().__init__()
        assert n_emb % n_head == 0
        self.q_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.kv_proj = nn.Linear(n_emb, 2 * n_emb, bias=False)
        self.c_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.resid_dropout = nn.Dropout(p_drop_attn)
        self.p_attn_dropout = p_drop_attn
        
        self.n_head = n_head
        self.n_emb = n_emb
        self.head_dim = n_emb // n_head

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        B, T, C = x.size()
        B_mem, T_mem, C_mem = memory.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if memory_kv_cache is not None:
            k, v = memory_kv_cache
        else:
            k, v = self.kv_proj(memory).split(self.n_emb, dim=2)
            k, v = map(lambda t: t.view(B_mem, T_mem, self.n_head, self.head_dim).transpose(1, 2), (k, v))

        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.p_attn_dropout if self.training else 0.0,
            is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ A single Transformer Decoder Block with self-attention, cross-attention, and MLP """
    def __init__(self, n_head: int, n_emb: int, p_drop_attn: float):
        super().__init__()
        self.ln_1 = RMSNorm(n_emb)
        self.attn = CausalSelfAttention(n_head, n_emb, p_drop_attn)
        self.ln_2 = RMSNorm(n_emb)
        self.cross_attn = CrossAttention(n_head, n_emb, p_drop_attn)
        self.ln_3 = RMSNorm(n_emb)
        self.mlp = MLP(n_emb, p_drop_attn)

    def forward(
        self, 
        x: torch.Tensor, 
        memory: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        memory_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        # Self-Attention
        attn_output, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + attn_output
        
        # Cross-Attention
        cross_attn_output = self.cross_attn(self.ln_2(x), memory, memory_kv_cache)
        x = x + cross_attn_output

        # MLP
        x = x + self.mlp(self.ln_3(x))
        return x, present


class AutoregressiveModel(ModuleAttrMixin):
    def __init__(self,
        vocab_size: int,
        max_seq_len: int,
        max_cond_len: int,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_emb = n_emb
        
        # Input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, n_emb)
        self.tok_pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, n_emb))
        self.cond_emb = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_len, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        self.encoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.Mish(),
            nn.Linear(4 * n_emb, n_emb)
        )
        
        self.blocks = nn.ModuleList([
            Block(n_head, n_emb, p_drop_attn) for _ in range(n_layer)
        ])
        
        # Decoder head
        self.ln_f = RMSNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.tok_emb.weight

        # Init
        self.init_weights_sp()

    def init_weights_sp(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, RMSNorm):
                nn.init.constant_(m.weight, 1.0)

    def forward(self, 
        tokens: torch.LongTensor, 
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for training. No KV cache is used here.
        tokens: (B, T_tok)
        cond: (B, T_cond, cond_dim) or (B, T_cond, emb_dim) if already encoded
        output: (B, T_tok, vocab_size)
        """
        T_tok = tokens.shape[1]
        T_cond = cond.shape[1]
        
        # Token processing
        tok_emb = self.tok_emb(tokens)
        pos_emb = self.tok_pos_emb[:, :T_tok, :]
        x = self.drop(tok_emb + pos_emb)

        # Condition processing
        cond_emb = self.cond_emb(cond)
        cond_pos_emb = self.cond_pos_emb[:, :T_cond, :]
        memory = self.drop(cond_emb + cond_pos_emb)
        memory = self.encoder(memory)

        # decoding
        for block in self.blocks:
            x, _ = block(x, memory)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def encode_cond(self,
        cond: torch.Tensor,
        precompute_memory_kv: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Encode the conditioning stream once for a whole decoding rollout.

        This is the exact conditioning pipeline used by both ``forward`` and
        ``generate``, factored out so that a multi-step (KV-cached) decoding
        loop can reuse it.  Unlike ``generate`` this is *not* wrapped in
        ``inference_mode``, so it is safe to use during training: gradients flow
        back into ``cond_emb`` / ``encoder`` / ``cross_attn.kv_proj``.

        cond: (B, T_cond, cond_dim)
        returns: (memory, memory_kv_cache)
        """
        T_cond = cond.shape[1]
        cond_emb = self.cond_emb(cond)
        cond_pos_emb = self.cond_pos_emb[:, :T_cond, :]
        memory = self.drop(cond_emb + cond_pos_emb)
        memory = self.encoder(memory)

        memory_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        if precompute_memory_kv:
            B_mem, T_mem = memory.shape[:2]
            memory_kv_cache = []
            for block in self.blocks:
                k_mem, v_mem = block.cross_attn.kv_proj(memory).split(self.n_emb, dim=2)
                k_mem, v_mem = map(lambda t:
                    t.view(B_mem, T_mem, self.n_head, self.n_emb // self.n_head).transpose(1, 2),
                    (k_mem, v_mem))
                memory_kv_cache.append((k_mem, v_mem))
        return memory, memory_kv_cache

    def forward_step(self,
        tokens: torch.LongTensor,
        memory: torch.Tensor,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        memory_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        pos_offset: int = 0,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """One (or a few) decoding step(s) with an explicit KV cache.

        Same math as ``forward``, but the conditioning stream is passed in
        pre-encoded and self-attention keys/values from earlier positions come
        from ``past_key_values`` instead of being recomputed.  Gradients flow
        through the cache, so a sequential training loop that accumulates
        per-step losses and backprops once is equivalent to a single pass over
        the same token sequence (up to per-step dropout resampling).

        tokens: (B, T_step) tokens for absolute positions
            [pos_offset, pos_offset + T_step)
        memory: (B, T_cond, n_emb) from ``encode_cond``
        pos_offset: absolute position of ``tokens[:, 0]`` in the sequence
        returns: (logits (B, T_step, vocab_size), updated past_key_values)
        """
        T_tok = tokens.shape[1]
        has_cache = past_key_values is not None and any(
            p is not None for p in past_key_values
        )
        if has_cache and T_tok > 1:
            # CausalSelfAttention disables causal masking whenever a cache is
            # present, so multi-token steps on top of a cache would leak future
            # positions.  Feed the prefix in one call, then one token per step.
            raise ValueError(
                "forward_step supports T_step > 1 only for the first (uncached) "
                f"call, got T_step={T_tok} with a non-empty KV cache."
            )
        tok_emb = self.tok_emb(tokens)
        pos_emb = self.tok_pos_emb[:, pos_offset:pos_offset + T_tok, :]
        x = self.drop(tok_emb + pos_emb)

        if past_key_values is None:
            past_key_values = [None] * self.n_layer
        new_past_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = \
            [None] * self.n_layer
        for i, block in enumerate(self.blocks):
            x, present = block(
                x, memory,
                layer_past=past_key_values[i],
                memory_kv_cache=None if memory_kv_cache is None else memory_kv_cache[i],
            )
            new_past_key_values[i] = present

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_past_key_values

    @torch.inference_mode()
    def generate(self,
        prefix: torch.LongTensor, 
        cond: torch.Tensor, 
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate tokens autoregressively with KV Caching.
        prefix: (B, T_pre)
        cond: (B, T_cond, cond_dim)
        max_new_tokens: int, max number of new tokens to generate
        temperature: float, sampling temperature
        top_k: Optional[int], if specified, use top-k sampling
        eos_id: Optional[int], if specified, stop generation when eos_id is generated
            all subsequent tokens will be set to eos_id
        output: (B, T_pre + max_new_tokens)
        """        
        # --- Pre-computation for condition ---
        T_cond = cond.shape[1]
        cond_emb = self.cond_emb(cond)
        cond_pos_emb = self.cond_pos_emb[:, :T_cond, :]
        memory = self.drop(cond_emb + cond_pos_emb)
        memory = self.encoder(memory)
        B_mem, T_mem = memory.shape[:2]
        
        # Pre-compute KV for cross-attention, as it's static
        memory_kv_cache = []
        for block in self.blocks:
            k_mem, v_mem = block.cross_attn.kv_proj(memory).split(self.n_emb, dim=2)
            k_mem, v_mem = map(lambda t: 
                t.view(B_mem, T_mem, self.n_head, self.n_emb // self.n_head).transpose(1, 2), 
                (k_mem, v_mem))
            memory_kv_cache.append((k_mem, v_mem))
            
        # --- Phase 1: Process prefix tokens to fill the KV cache ---
        T_pre = prefix.shape[1]
        tok_emb = self.tok_emb(prefix)
        pos_emb = self.tok_pos_emb[:, :T_pre, :]
        x = self.drop(tok_emb + pos_emb)

        past_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.n_layer
        for i, block in enumerate(self.blocks):
            x, present = block(x, memory, layer_past=None, memory_kv_cache=memory_kv_cache[i])
            past_key_values[i] = present

        # Get logits for the very next token
        x = self.ln_f(x[:, -1:, :]) # Only need the last token's output
        logits = self.head(x)
        
        # --- Phase 2: Autoregressively generate new tokens ---
        out_tokens = prefix
        finished = torch.zeros(B_mem, dtype=torch.bool, device=prefix.device) if eos_id is not None else None
        for i in range(max_new_tokens):
            # Sample the next token
            if temperature > 0:
                logits = logits.squeeze(1) / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)  # [B, 1]
            
            # if an EOS token is generated, mark the sequence as finished
            # and replace all subsequent tokens with EOS
            if eos_id is not None:
                if finished.any():
                    next_token = torch.where(
                        finished.view(-1, 1),
                        torch.full_like(next_token, eos_id),
                        next_token
                    )
                finished = finished | (next_token.squeeze(-1) == eos_id)

            out_tokens = torch.cat((out_tokens, next_token), dim=1)

            # If all sequences are finished, stop early
            if eos_id is not None and finished.all():
                break
            
            # If we're done, break
            if out_tokens.shape[1] >= T_pre + max_new_tokens:
                break
                
            # Prepare input for the next step (only the new token)
            tok_emb = self.tok_emb(next_token)
            current_pos = T_pre + i
            pos_emb = self.tok_pos_emb[:, current_pos:current_pos+1, :]
            x = self.drop(tok_emb + pos_emb)

            # Forward pass with KV cache
            for layer_idx, block in enumerate(self.blocks):
                x, present = block(x, memory, 
                    layer_past=past_key_values[layer_idx], 
                    memory_kv_cache=memory_kv_cache[layer_idx]
                )
                past_key_values[layer_idx] = present # Update cache

            x = self.ln_f(x)
            logits = self.head(x)
        
        return out_tokens
