import torch
import torch.nn.functional as F
from typing import Tuple, Union, List, Optional

from oat.model.common.normalizer import LinearNormalizer 
from oat.tokenizer.base_tokenizer import BaseTokenizer 
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.quantizer.fsq import FSQ

def pad_token_seq(token_ids: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """
    Pad the token sequence to the maximum length.

    Args:
        token_ids: The token id sequence of shape [B, l].
        max_seq_len: The maximum sequence length L.

    Returns:
        Padded token id sequence.
    """
    device, dtype = token_ids.device, token_ids.dtype
    pad_len = max_seq_len - token_ids.shape[1]
    pad_seq = torch.zeros((token_ids.shape[0], pad_len), device=device, dtype=dtype)
    return torch.cat([token_ids, pad_seq], dim=1)  # [B, L]


class OATTok(BaseTokenizer):
    def __init__(self,
        encoder: RegisterEncoder,
        decoder: SinglePassDecoder,
        quantizer: FSQ,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.normalizer = LinearNormalizer()
        self.latent_horizon = self.decoder.latent_horizon
        self.token_seq_len = (
            getattr(self.quantizer, "token_seq_len", None)
            or self.latent_horizon * getattr(self.quantizer, "num_stages", 1)
        )

    def _token_lens_to_eval_keep_k(self, token_lens: List[int]) -> List[int]:
        return [min(token_len, self.latent_horizon) for token_len in token_lens]

    def get_optimizer(
        self, 
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """Create an AdamW optimizer with weight decay for 2D parameters only."""
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch) -> torch.Tensor:
        samples = batch['action']
        
        # normalize
        nsamples = self.normalizer['action'].normalize(samples)
        
        # encode & quantize
        latents = self.encoder(nsamples)
        latents, _ = self.quantizer(latents)

        # decode
        recons = self.decoder(latents)
        loss = F.mse_loss(recons, nsamples)

        return loss

    def encode(self, samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # samples: (B, T, sample_dim)

        # normalize
        nsamples = self.normalizer['action'].normalize(samples)
        # encode & quantize
        latents = self.encoder(nsamples)
        latents, tokens = self.quantizer(latents)
        return latents, tokens

    def decode(self, 
        latents: torch.Tensor, 
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # latents: (B, T', encoder_emb_dim)
        if eval_keep_k is None:
            eval_keep_k = [latents.shape[1]] * latents.shape[0]

        assert all([k <= self.latent_horizon for k in eval_keep_k]), \
            f"All eval_keep_k must be <= {self.latent_horizon}"

        # decode - returns normalized samples
        nsamples = self.decoder(latents, eval_keep_k=eval_keep_k)

        # unnormalize
        samples = self.normalizer['action'].unnormalize(nsamples)
        return samples

    def autoencode(self, 
        samples: torch.Tensor, 
        eval_keep_k: Optional[List[int]] = None, 
    ) -> torch.Tensor:
        # samples: (B, T, sample_dim)
        latents, _ = self.encode(samples)
        recons = self.decode(latents, eval_keep_k=eval_keep_k)
        return recons

    def tokenize(self, samples: torch.Tensor) -> torch.Tensor:
        # samples: (B, T, sample_dim)
        _, tokens = self.encode(samples)
        return tokens

    def detokenize(self, 
        tokens: Union[torch.Tensor, List[List[int]]],
        token_lens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # tokens: (B, T') or list of list of int
        
        # standardize
        if isinstance(tokens, list):
            rows = []
            inferred_lens = []
            for token_row in tokens:
                if isinstance(token_row, torch.Tensor):
                    row = token_row
                    if row.ndim == 2:
                        assert row.shape[0] == 1, "List tensor entries must have batch size 1."
                        row = row.squeeze(0)
                else:
                    row = torch.tensor(token_row, dtype=torch.long)
                if row.shape[-1] > self.token_seq_len:
                    raise ValueError(f"Expected at most {self.token_seq_len} tokens, got {row.shape[-1]}.")
                inferred_lens.append(row.shape[-1])
                rows.append(pad_token_seq(row.unsqueeze(0), self.token_seq_len))
            token_lens = inferred_lens if token_lens is None else token_lens
            tokens = torch.cat(rows, dim=0)
        elif isinstance(tokens, torch.Tensor):
            if tokens.ndim == 1:
                tokens = tokens.unsqueeze(0)
            token_lens = [tokens.shape[1]] * tokens.shape[0] if token_lens is None else token_lens
            if tokens.shape[-1] > self.token_seq_len:
                raise ValueError(f"Expected at most {self.token_seq_len} tokens, got {tokens.shape[-1]}.")
            if tokens.shape[-1] < self.token_seq_len:
                tokens = pad_token_seq(tokens, self.token_seq_len)
        else:
            raise ValueError(f'Unknown token type {type(tokens)}')

        # codebook lookup & decode
        latents = self.quantizer.indices_to_embedding(tokens, token_lens=token_lens)
        eval_keep_k = self._token_lens_to_eval_keep_k(token_lens)
        samples = self.decode(latents, eval_keep_k=eval_keep_k)
        return samples
