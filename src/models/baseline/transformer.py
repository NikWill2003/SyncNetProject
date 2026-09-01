from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from ...core.config import ModelConfig
from ...core.contracts import VQABatch, VQAOutput
from ..common.img_enc import build_image_encoder
from ..common.pos_enc import PositionalEncoder1D, PositionalEncoder2D
from ..common.qst_enc import build_question_encoder


@dataclass
class VQATransformerConfig(ModelConfig):
    name: str = 'transformer'

    q_encoder: dict[str, Any] = field(default_factory=lambda: {'name': 'identity'})
    encoder: dict[str, Any] = field(default_factory=lambda: {
        'name': 'patchify',
        'ch': 128,
        'patch_size': 5,
    })

    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 2
    dropout: float = 0.0

    pos_enc: str = 'learnt_2d'                  # learnt_1d | learnt_2d
    q_conditioning: str = 'token'               # film | broadcast_cat | token | token_seq
    share_layer_weights: bool = False
    readout: str = 'cls'                        # cls | mean | flatten


class VQATransformer(nn.Module):

    allowed_q_encoders = {'identity', 'mlp', 'lstm', 'tokenise'}

    def __init__(self, cfg: VQATransformerConfig, dataset: str, answer_dim: int) -> None:
        super().__init__()

        if cfg.hidden_dim % cfg.n_heads != 0:
            raise ValueError(f'hidden_dim ({cfg.hidden_dim}) must be divisible by n_heads ({cfg.n_heads})')

        if cfg.readout not in ('cls', 'mean', 'flatten'):
            raise ValueError(f'readout must be cls or mean or flatten, got {cfg.readout!r}')

        if cfg.q_conditioning not in ('film', 'broadcast_cat', 'token', 'token_seq'):
            raise ValueError(f'unknown question conditioning: {cfg.q_conditioning!r}')

        self.question_encoder = build_question_encoder(cfg.q_encoder, dataset, allowed=self.allowed_q_encoders)
        self.encoder = build_image_encoder(cfg.encoder, dataset)

        self.n_layers = cfg.n_layers
        self.q_conditioning = cfg.q_conditioning
        self.share_layer_weights = cfg.share_layer_weights
        self.readout = cfg.readout

        if cfg.q_conditioning == 'token_seq' and len(self.question_encoder.output_shape) != 2:
            raise ValueError(
                "q_conditioning='token_seq' requires a question encoder that returns a token sequence"
            )

        self.build_question_conditioning(cfg)
        self.build_positional_encoding(cfg)
        self.build_layers(cfg)

        head_dim = cfg.hidden_dim * self.encoder.n_tokens if cfg.readout == 'flatten' else cfg.hidden_dim

        self.head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, answer_dim),
        )

    def build_question_conditioning(self, cfg: VQATransformerConfig) -> None:
        image_dim = self.encoder.ch
        question_shape = self.question_encoder.output_shape
        question_dim = math.prod(question_shape)

        if cfg.q_conditioning in ('token', 'token_seq', 'film'):
            self.image_proj = nn.Linear(image_dim, cfg.hidden_dim) if image_dim != cfg.hidden_dim else nn.Identity()

        if cfg.q_conditioning == 'token':
            self.question_proj = nn.Linear(question_dim, cfg.hidden_dim)
            self.question_pos = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)

        elif cfg.q_conditioning == 'token_seq':
            if len(question_shape) != 2:
                raise ValueError(
                    'questions shape should be two dimensional when using token_seq quesition conditioning'
                    )
            
            question_len, question_token_dim = question_shape
            self.question_proj = nn.Linear(question_token_dim, cfg.hidden_dim)
            self.question_pos = nn.Parameter(torch.randn(1, question_len, cfg.hidden_dim) * 0.02)

        elif cfg.q_conditioning == 'broadcast_cat':
            question_proj_dim = cfg.hidden_dim - image_dim

            if question_proj_dim <= 0:
                raise ValueError(
                    f'encoder.ch ({image_dim}) must be smaller than hidden_dim ({cfg.hidden_dim}) '
                    'for q_conditioning="broadcast_cat"'
                )

            self.question_proj = nn.Linear(question_dim, question_proj_dim)

        elif cfg.q_conditioning == 'film':
            self.film_gamma = nn.Linear(question_dim, image_dim)
            self.film_beta = nn.Linear(question_dim, image_dim)

    def build_positional_encoding(self, cfg: VQATransformerConfig) -> None:
        if cfg.pos_enc == 'learnt_1d':
            self.pos_enc = PositionalEncoder1D(cfg.hidden_dim, self.encoder.n_tokens)

        elif cfg.pos_enc == 'learnt_2d':
            self.pos_enc = PositionalEncoder2D(cfg.hidden_dim, self.encoder.spatial, self.encoder.spatial)

        else:
            raise ValueError(f'unknown positional encoding: {cfg.pos_enc!r}')

        self.cls = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)

        self.n_prefix = 1

        if cfg.q_conditioning == 'token':
            self.n_prefix += 1

        elif cfg.q_conditioning == 'token_seq':
            self.n_prefix += self.question_encoder.output_shape[0]

    def build_layers(self, cfg: VQATransformerConfig) -> None:
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
            self.transformer = layer
        else:
            self.transformer = nn.TransformerEncoder(
                layer,
                num_layers=cfg.n_layers,
                enable_nested_tensor=False,
            )

    def make_token_sequence(self, images: Tensor, questions: Tensor) -> Tensor:
        batch_size = images.size(0)

        question = self.question_encoder(questions)
        question_flat = question.flatten(1)
        features = self.encoder(images)

        if self.q_conditioning == 'film':
            gamma = self.film_gamma(question_flat)[..., None, None]
            beta = self.film_beta(question_flat)[..., None, None]
            features = features * (1.0 + gamma) + beta

        tokens = features.flatten(2).transpose(1, 2)
        cls = self.cls.expand(batch_size, -1, -1)

        if self.q_conditioning == 'token':
            tokens = self.pos_enc(self.image_proj(tokens))
            question_token = self.question_proj(question_flat).unsqueeze(1) + self.question_pos
            return torch.cat([cls, question_token, tokens], dim=1)

        if self.q_conditioning == 'token_seq':
            tokens = self.pos_enc(self.image_proj(tokens))
            question_tokens = self.question_proj(question) + self.question_pos
            return torch.cat([cls, question_tokens, tokens], dim=1)

        if self.q_conditioning == 'film':
            tokens = self.pos_enc(self.image_proj(tokens))
            return torch.cat([cls, tokens], dim=1)

        question_broadcast = self.question_proj(question_flat).unsqueeze(1).expand(-1, tokens.size(1), -1)
        tokens = self.pos_enc(torch.cat([tokens, question_broadcast], dim=-1))

        return torch.cat([cls, tokens], dim=1)

    def run_layers(self, tokens: Tensor) -> Tensor:
        if self.share_layer_weights:
            for _ in range(self.n_layers):
                tokens = self.transformer(tokens)
        else:
            tokens = self.transformer(tokens)

        return tokens

    def compute_readout(self, tokens: Tensor) -> Tensor:
        if self.readout == 'cls':
            representation = tokens[:, 0]

        else:
            patches = tokens[:, self.n_prefix:]

            if self.readout == 'mean':
                representation = patches.mean(dim=1)
            else:
                representation = patches.flatten(1)

        return self.head(representation)

    def forward(self, batch: VQABatch, **overrides) -> VQAOutput:
        tokens = self.make_token_sequence(batch['images'], batch['questions'])
        tokens = self.run_layers(tokens)

        return {'logits': self.compute_readout(tokens)}

    @classmethod
    def from_config(
            cls, model_cfg: VQATransformerConfig, dataset: str, answer_dim: int
        ) -> VQATransformer:

        return cls(model_cfg, dataset, answer_dim)