from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .encoders import PatchifyEncoder
from .pos_enc import PositionalEncoder1D, PositionalEncoder2D
from .question_encoders import QuestionEncoder

QCond = Literal['film', 'broadcast_cat', 'token', 'token_seq']

# Names used before the shared-model migration. Kept only to produce a
# useful error: silently aliasing them would let a config sweep think it
# had covered an arm it had not.
_RENAMED = {
    'token_proj': 'token',
    'token_emb': 'token_seq',
}


class VQATransformer(nn.Module):
    """Patch-token transformer over (image, question).

    Conditioning axis -- WHERE the question enters. The Sort-of-CLEVR
    sweep found this worth ~32 points of ternary accuracy at matched
    capacity (late `token` 0.547 vs per-token `broadcast_cat` 0.865),
    and a 3x larger late-conditioned model was worse still, so it is an
    axis rather than a fixed choice:

        film           modulate patch features before tokenising
        broadcast_cat  concatenate q onto every visual token
        token          flattened q as one extra sequence position
        token_seq      the question's L positions kept separate

    On Sort-of-CLEVR the question encoder has L = 1, so `token_seq`
    degenerates to `token`; it exists for SQOOP, where [x, rel, y] is
    genuinely three tokens.
    """

    def __init__(
        self,
        q_encoder: QuestionEncoder,
        img_size: int,
        answer_size: int,
        patch_size: int,
        patch_emb_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        ffn_mult: int,
        dropout: float,
        pos_enc: Literal['learnt_1d', 'learnt_2d'],
        q_conditioning: QCond,
        share_layer_weights: bool,
        **kwargs,
    ) -> None:
        super().__init__()

        if q_conditioning in _RENAMED:
            raise ValueError(
                f'q_conditioning={q_conditioning!r} was renamed to '
                f'{_RENAMED[q_conditioning]!r} in the shared-model '
                'migration; update the config rather than relying on an '
                'alias, so sweeps record the arm they actually ran.'
            )

        self.n_layers = n_layers
        self.q_conditioning = q_conditioning
        self.share_layer_weights = share_layer_weights
        self.q_encoder = q_encoder

        if hidden_dim % n_heads != 0:
            raise ValueError(
                f'hidden_dim ({hidden_dim}) must be divisible by '
                f'n_heads ({n_heads}).'
            )
        if img_size % patch_size != 0:
            raise ValueError(
                f'img_size ({img_size}) must be divisible by '
                f'patch_size ({patch_size}).'
            )

        self.encoder = PatchifyEncoder(img_size, patch_emb_dim, patch_size)

        q_dim = q_encoder.out_dim

        if q_conditioning in ('token', 'token_seq', 'film'):
            self.tok_proj = (
                nn.Linear(patch_emb_dim, hidden_dim)
                if patch_emb_dim != hidden_dim
                else nn.Identity()
            )

        if q_conditioning == 'token':
            self.q_emb = nn.Linear(q_dim, hidden_dim)
            self.q_pos_enc = nn.Parameter(
                torch.randn(1, 1, hidden_dim) * 0.02
            )

        elif q_conditioning == 'token_seq':
            self.q_emb = nn.Linear(q_encoder.emb_dim, hidden_dim)
            self.q_pos_enc = nn.Parameter(
                torch.randn(1, q_encoder.seq_len, hidden_dim) * 0.02
            )

        elif q_conditioning == 'broadcast_cat':
            q_emb_dim = hidden_dim - patch_emb_dim
            if q_emb_dim <= 0:
                raise ValueError(
                    f'patch_emb_dim ({patch_emb_dim}) must be smaller '
                    f'than hidden_dim ({hidden_dim}) when using '
                    'q_conditioning="broadcast_cat".'
                )
            self.q_emb = nn.Linear(q_dim, q_emb_dim)

        elif q_conditioning == 'film':
            self.film_gamma = nn.Linear(q_dim, patch_emb_dim)
            self.film_beta = nn.Linear(q_dim, patch_emb_dim)

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
            raise ValueError(f'Unknown positional encoding: {pos_enc}')

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
            self.transformer = layer
        else:
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

    def forward(self, images: Tensor, questions: Tensor) -> Tensor:
        B = images.size(0)
        q_seq = self.q_encoder(questions)          # (B, L, D)
        q_flat = q_seq.flatten(1)                  # (B, L*D)

        feats = self.encoder(images)               # (B, C, H, W)

        if self.q_conditioning == 'film':
            gamma = self.film_gamma(q_flat).unsqueeze(-1).unsqueeze(-1)
            beta = self.film_beta(q_flat).unsqueeze(-1).unsqueeze(-1)
            feats = feats * (1.0 + gamma) + beta

        tokens = feats.flatten(2).transpose(1, 2)  # (B, T, C)
        cls = self.cls.expand(B, -1, -1)

        if self.q_conditioning == 'token':
            tokens = self.pos_enc(self.tok_proj(tokens))
            q = self.q_emb(q_flat).unsqueeze(1) + self.q_pos_enc
            x = torch.cat([cls, q, tokens], dim=1)

        elif self.q_conditioning == 'token_seq':
            tokens = self.pos_enc(self.tok_proj(tokens))
            q = self.q_emb(q_seq) + self.q_pos_enc
            x = torch.cat([cls, q, tokens], dim=1)

        elif self.q_conditioning == 'film':
            # the question is already inside the features
            tokens = self.pos_enc(self.tok_proj(tokens))
            x = torch.cat([cls, tokens], dim=1)

        else:                                       # broadcast_cat
            q = self.q_emb(q_flat).unsqueeze(1).expand(
                -1, tokens.size(1), -1
            )
            tokens = self.pos_enc(torch.cat([tokens, q], dim=-1))
            x = torch.cat([cls, tokens], dim=1)

        if self.share_layer_weights:
            for _ in range(self.n_layers):
                x = self.transformer(x)
        else:
            x = self.transformer(x)

        return self.head(x[:, 0])
