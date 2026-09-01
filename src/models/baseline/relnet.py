from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.config import ModelConfig
from ...core.contracts import VQABatch, VQAOutput
from ..common.img_enc import build_image_encoder
from ..common.qst_enc import build_question_encoder


@dataclass
class VQARelNetConfig(ModelConfig):
    name: str = 'relnet'

    q_encoder: dict[str, Any] = field(default_factory=lambda: {'name': 'identity'})

    g_hidden_dim: int = 256
    f_hidden_dim: int = 256

    pair_spatial: int | None = None
    pair_pool: str = 'max' # max or avg

    encoder: dict[str, Any] = field(default_factory=lambda: {'name': 'cnn'})


class VQARelNet(nn.Module):

    allowed_q_encoders = {'identity', 'mlp', 'lstm'}

    def __init__(self, cfg: VQARelNetConfig, dataset: str, answer_dim: int) -> None:
        super().__init__()

        if cfg.pair_pool not in ('max', 'avg'):
            raise ValueError(f'pair_pool must be max|avg, got {cfg.pair_pool!r}')

        self.question_encoder = build_question_encoder(cfg.q_encoder, dataset, self.allowed_q_encoders)
        self.encoder = build_image_encoder(cfg.encoder, dataset)

        question_shape = self.question_encoder.output_shape
        if len(question_shape) != 1:
            raise ValueError(f'VQARelNet requires a vector question representation, got {question_shape}')

        question_dim = question_shape[0]

        self.pair_spatial = cfg.pair_spatial
        self.pair_pool = cfg.pair_pool

        object_dim = self.encoder.ch + 2

        self.g = nn.Sequential(
            nn.Linear(2 * object_dim + question_dim, cfg.g_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.g_hidden_dim, cfg.g_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.g_hidden_dim, cfg.g_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.g_hidden_dim, cfg.g_hidden_dim),
            nn.ReLU(),
        )

        self.f = nn.Sequential(
            nn.Linear(cfg.g_hidden_dim, cfg.f_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.f_hidden_dim, cfg.f_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.f_hidden_dim, answer_dim),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:

        if self.pair_spatial is None:
            return x

        output_size = (self.pair_spatial, self.pair_spatial)

        if self.pair_pool == 'max':
            return F.adaptive_max_pool2d(x, output_size)

        return F.adaptive_avg_pool2d(x, output_size)


    def forward(self, batch: VQABatch, **overrides) -> VQAOutput:
        image = self._pool(self.encoder(batch['images']))
        question = self.question_encoder(batch['questions'])

        B, _, H, W = image.shape
        N = H * W

        # spatial cells become objects.
        objects = image.flatten(2).transpose(1, 2)

        # append normalized spatial coordinates.
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=image.device),
            torch.linspace(-1.0, 1.0, W, device=image.device),
            indexing='ij',
        )

        coords = torch.stack([x, y], dim=-1).reshape(1, N, 2).expand(B, -1, -1)
        objects = torch.cat([objects, coords], dim=-1)

        # construct all ordered object pairs.
        o_i = objects[:, :, None, :].expand(-1, -1, N, -1)
        o_j = objects[:, None, :, :].expand(-1, N, -1, -1)
        q = question[:, None, None, :].expand(-1, N, N, -1)

        relations = self.g(torch.cat([o_i, o_j, q], dim=-1))
        relations = relations.sum(dim=(1, 2))

        return {'logits': self.f(relations)}

    @classmethod
    def from_config(
            cls, model_cfg: VQARelNetConfig, dataset: str,answer_dim: int,
        ) -> VQARelNet:
        
        return cls(model_cfg, dataset, answer_dim)