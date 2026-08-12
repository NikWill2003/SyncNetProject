from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ....core.config import ModelConfig
from ....core.encoders import PatchifyEncoder
from ....core.pos_enc import PositionalEncoder1D, PositionalEncoder2D
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput
from ..data import constants as C

@dataclass
class SortOfClevrTransformerConfig(ModelConfig):
    name: str = 'sort_of_clevr_transformer'
    # encoder
    patch_emb_dim: int = 128
    patch_size: int = 5

    # transformer
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 2
    dropout: float = 0.0

    # positional encoding
    pos_enc: str = 'learnt_2d' # learnt_1d | learnt_2d

    # where the question enters
    q_conditioning: str = 'token' # token | broadcast_cat | film

    share_layer_weights: bool = False # false -> normal transformer, true -> looped

class SortOfClevrTransformer(nn.Module):
    has_rotors = False
    is_syncnet = False

    def __init__(
        self,
        cfg: SortOfClevrTransformerConfig,
        data_cfg: SortOfClevrDataConfig,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.data_cfg = data_cfg

        if cfg.hidden_dim % cfg.n_heads != 0:
            raise ValueError(
                f'hidden_dim ({cfg.hidden_dim}) must be divisible by n_heads ({cfg.n_heads}).'
            )

        if data_cfg.img_size % cfg.patch_size != 0:
            raise ValueError(
                f'img_size ({data_cfg.img_size}) must be divisible by patch_size ({cfg.patch_size}).'
            )

        self.encoder = PatchifyEncoder(
            data_cfg.img_size,
            cfg.patch_emb_dim,
            cfg.patch_size,
        )

        # token projection: spatial_emd.dim != hidden_dim
        if cfg.q_conditioning in ('token', 'film'):
            self.tok_proj = (
                nn.Linear(cfg.patch_emb_dim, cfg.hidden_dim)
                if cfg.patch_emb_dim != cfg.hidden_dim
                else nn.Identity()
            )

        if cfg.q_conditioning == 'token':
            # question is one token
            self.q_enc = nn.Linear(
                C.QUESTION_SIZE,
                cfg.hidden_dim,
            )

            self.q_pos_enc = nn.Parameter(
                torch.randn(1, 1, cfg.hidden_dim) * 0.02
            )

        elif cfg.q_conditioning == 'broadcast_cat':
            # q representation is concatenated onto every visual token
            q_emb_dim = cfg.hidden_dim - cfg.patch_emb_dim

            if q_emb_dim <= 0:
                raise ValueError(
                    f'patch_emb_dim ({cfg.patch_emb_dim}) must be smaller '
                    f'than hidden_dim ({cfg.hidden_dim}) when using '
                    f'q_conditioning="broadcast_cat".'
                )

            self.q_enc = nn.Linear(
                C.QUESTION_SIZE,
                q_emb_dim,
            )

        elif cfg.q_conditioning == 'film':
            # feature wise modulation to patch embeddings conditioned on the question
            self.film_gamma = nn.Linear(C.QUESTION_SIZE, cfg.patch_emb_dim)
            self.film_beta = nn.Linear(C.QUESTION_SIZE, cfg.patch_emb_dim)

        else:
            raise ValueError(
                f'Unknown question conditioning: {cfg.q_conditioning}'
            )

        if cfg.pos_enc == 'learnt_1d':
            self.pos_enc = PositionalEncoder1D(
                cfg.hidden_dim, self.encoder.n_tokens
            )
        elif cfg.pos_enc == 'learnt_2d':
            self.pos_enc = PositionalEncoder2D(
                cfg.hidden_dim, self.encoder.spatial, self.encoder.spatial
            )
        else:
            raise ValueError(
                f'Unknown positional encoding: {cfg.pos_enc}'
            )

        self.cls = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim * cfg.ffn_mult,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        if cfg.share_layer_weights:
            # one layer repeatedly applied n_layers times
            self.transformer = layer
        else:
            # n_layers independently parameterised layers
            self.transformer = nn.TransformerEncoder(
                layer,
                num_layers=cfg.n_layers,
                enable_nested_tensor=False,
            )

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, C.ANSWER_SIZE),
        )

    def forward(
        self,
        images: Tensor,
        questions: Tensor,
        **kwards,
    ) -> SortOfClevrOutput:

        B = images.size(0)
        q_raw = questions.float()

        # (B, C, H, W)
        feats = self.encoder(images)

        if self.cfg.q_conditioning == 'film':
            gamma = self.film_gamma(q_raw).unsqueeze(-1).unsqueeze(-1)
            beta = self.film_beta(q_raw).unsqueeze(-1).unsqueeze(-1)
            feats = feats * (1.0 + gamma) + beta

        # (B, C, H, W) -> (B, C, T) -> (B, T, C)
        tokens = feats.flatten(2).transpose(1, 2)
        cls = self.cls.expand(B, -1, -1)

        if self.cfg.q_conditioning == 'token':

            tokens = self.tok_proj(tokens)
            tokens = self.pos_enc(tokens)
            q = self.q_enc(q_raw).unsqueeze(1) + self.q_pos_enc
            x = torch.cat([cls, q, tokens], dim=1)

        elif self.cfg.q_conditioning == 'film':
            # the question is already inside the features
            tokens = self.tok_proj(tokens)
            tokens = self.pos_enc(tokens)
            x = torch.cat([cls, tokens], dim=1)

        else:
            q = self.q_enc(q_raw) # (B, q_emb_dim)
            q = q.unsqueeze(1).expand(
                -1, tokens.size(1), -1,
            ) # (B, T, q_emb_dim)

            # patch embedding | question embedding -> (B, T, H)
            tokens = torch.cat([tokens, q], dim=-1)
            tokens = self.pos_enc(tokens)
            x = torch.cat([cls, tokens], dim=1)

        if self.cfg.share_layer_weights:
            for _ in range(self.cfg.n_layers):
                x = self.transformer(x)
        else:
            x = self.transformer(x)

        return {'logits': self.head(x[:, 0])}

    @classmethod
    def from_config(
        cls,
        cfg: SortOfClevrTransformerConfig,
        data_cfg: SortOfClevrDataConfig,
    ) -> SortOfClevrTransformer:
        return cls(cfg=cfg, data_cfg=data_cfg)
