from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .encoders import PatchifyEncoder
from .pos_enc import PositionalEncoder1D, PositionalEncoder2D

class VQATransformer(nn.Module):

    def __init__(
        self,
        img_size,
        question_size,
        answer_size,
        patch_size,
        patch_emb_dim,
        hidden_dim,
        n_heads,
        n_layers,
        ffn_mult,
        dropout,
        pos_enc: Literal['learnt_1d', 'learnt_2d'],
        q_conditioning: Literal['token_proj', 'token_emb','broadcast_cat', 'film'],
        share_layer_weights: bool,
        vocab_size: int | None=None,
        **kwargs
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.q_conditioning = q_conditioning
        self.share_layer_weights = share_layer_weights

        if hidden_dim % n_heads != 0:
            raise ValueError(
                f'hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads}).'
            )

        if img_size % patch_size != 0:
            raise ValueError(
                f'img_size ({img_size}) must be divisible by patch_size ({patch_size}).'
            )

        self.encoder = PatchifyEncoder(
            img_size,
            patch_emb_dim,
            patch_size,
        )

        # token projection: spatial_emd.dim != hidden_dim
        if q_conditioning in ('token_proj', 'token_emb', 'film'):
            self.tok_proj = (
                nn.Linear(patch_emb_dim, hidden_dim)
                if patch_emb_dim != hidden_dim
                else nn.Identity()
            )
    

        if q_conditioning == 'token_proj':
            # question is one token
            self.q_emb = nn.Linear(
                question_size,
                hidden_dim,
            )

            self.q_pos_enc = nn.Parameter(
                torch.randn(1, 1, hidden_dim) * 0.02
            )

        elif q_conditioning == 'token_emb':
            if vocab_size is None:
                raise ValueError('vocab size is required to use question token embeddings')
            
            self.q_emb = nn.Embedding(
                vocab_size, hidden_dim
            )
            self.q_pos_enc = nn.Parameter(
                torch.randn(1, 3, hidden_dim) * 0.02
                )

        elif q_conditioning == 'broadcast_cat':
            # q representation is concatenated onto every visual token
            q_emb_dim = hidden_dim - patch_emb_dim

            if q_emb_dim <= 0:
                raise ValueError(
                    f'patch_emb_dim ({patch_emb_dim}) must be smaller '
                    f'than hidden_dim ({hidden_dim}) when using '
                    f'q_conditioning="broadcast_cat".'
                )

            self.q_emb = nn.Linear(
                question_size,
                q_emb_dim,
            )

        elif q_conditioning == 'film':
            # feature wise modulation to patch embeddings conditioned on the question
            self.film_gamma = nn.Linear(question_size, patch_emb_dim)
            self.film_beta = nn.Linear(question_size, patch_emb_dim)

        else:
            raise ValueError(
                f'Unknown question conditioning: {q_conditioning}'
            )

        if pos_enc == 'learnt_1d':
            self.pos_enc = PositionalEncoder1D(
                hidden_dim, self.encoder.n_tokens
            )
        elif pos_enc == 'learnt_2d':
            self.pos_enc = PositionalEncoder2D(
                hidden_dim, self.encoder.spatial, self.encoder.spatial
            )
        else:
            raise ValueError(
                f'Unknown positional encoding: {pos_enc}'
            )

        self.cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        if share_layer_weights:
            # one layer repeatedly applied n_layers times
            self.transformer = layer
        else:
            # n_layers independently parameterised layers
            self.transformer = nn.TransformerEncoder(
                layer,
                num_layers=n_layers,
                enable_nested_tensor=False,
            )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, answer_size),
        )

    def forward(
        self,
        images: Tensor,
        questions: Tensor,
    ) -> Tensor:

        B = images.size(0)

        # (B, C, H, W)
        feats = self.encoder(images)

        if self.q_conditioning == 'film':
            q_raw = questions.float()
            gamma = self.film_gamma(q_raw).unsqueeze(-1).unsqueeze(-1)
            beta = self.film_beta(q_raw).unsqueeze(-1).unsqueeze(-1)
            feats = feats * (1.0 + gamma) + beta

        # (B, C, H, W) -> (B, C, T) -> (B, T, C)
        tokens = feats.flatten(2).transpose(1, 2)
        cls = self.cls.expand(B, -1, -1)

        if self.q_conditioning == 'token_proj':

            q_raw = questions.float()
            tokens = self.tok_proj(tokens)
            tokens = self.pos_enc(tokens)
            q = self.q_emb(q_raw).unsqueeze(1) + self.q_pos_enc
            x = torch.cat([cls, q, tokens], dim=1)

        elif self.q_conditioning == 'token_emb':

            q_raw = questions.long()
            tokens = self.tok_proj(tokens)
            tokens = self.pos_enc(tokens)
            q = self.q_emb(q_raw) + self.q_pos_enc
            x = torch.cat([cls, q, tokens], dim=1)

        elif self.q_conditioning == 'film':
            # the question is already inside the features
            tokens = self.tok_proj(tokens)
            tokens = self.pos_enc(tokens)
            x = torch.cat([cls, tokens], dim=1)

        else:
            q_raw = questions.float()
            q = self.q_emb(q_raw) # (B, q_emb_dim)
            q = q.unsqueeze(1).expand(
                -1, tokens.size(1), -1,
            ) # (B, T, q_emb_dim)

            # patch embedding | question embedding -> (B, T, H)
            tokens = torch.cat([tokens, q], dim=-1)
            tokens = self.pos_enc(tokens)
            x = torch.cat([cls, tokens], dim=1)

        if self.share_layer_weights:
            for _ in range(self.n_layers):
                x = self.transformer(x)
        else:
            x = self.transformer(x)

        return self.head(x[:, 0])