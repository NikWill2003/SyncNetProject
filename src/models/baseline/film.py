from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ...core.config import ModelConfig
from ...core.contracts import VQABatch, VQAOutput
from ..common.img_enc import build_image_encoder
from ..common.qst_enc import build_question_encoder


class FiLMedBlock(nn.Module):

    def __init__(self, ch: int, n_extra: int = 0) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(ch + n_extra, ch, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm = nn.BatchNorm2d(ch, affine=False)

    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor, extra: Tensor | None = None) -> Tensor:
        if extra is not None:
            x = torch.cat([x, extra], dim=1)

        h = torch.relu(self.conv1(x))
        residual = h

        h = self.norm(self.conv2(h))
        h = h * (1.0 + gamma[..., None, None]) + beta[..., None, None]

        return torch.relu(h) + residual


@dataclass
class VQAFiLMConfig(ModelConfig):
    name: str = 'film'

    q_encoder: dict[str, Any] = field(default_factory=lambda: {'name': 'identity'})

    n_blocks: int = 4
    block_ch: int = 128
    classifier_ch: int = 512
    classifier_hidden: int = 1024
    use_coords: bool = True

    encoder: dict[str, Any] = field(default_factory=lambda: {'name': 'cnn'})


class VQAFiLM(nn.Module):

    allowed_q_encoders = {'identity', 'mlp', 'lstm'}

    def __init__(self, cfg: VQAFiLMConfig, dataset: str, answer_dim: int) -> None:
        super().__init__()

        self.question_encoder = build_question_encoder(cfg.q_encoder, dataset, self.allowed_q_encoders)
        self.encoder = build_image_encoder(cfg.encoder, dataset)

        question_shape = self.question_encoder.output_shape
        if len(question_shape) != 1:
            raise ValueError(f'VQAFiLM requires a vector question representation, got {question_shape}')

        question_dim = question_shape[0]

        self.n_blocks = cfg.n_blocks
        self.block_ch = cfg.block_ch

        self.proj = nn.Conv2d(self.encoder.ch, cfg.block_ch, 1) if self.encoder.ch != cfg.block_ch else nn.Identity()

        n_extra = 2 if cfg.use_coords else 0
        self.blocks = nn.ModuleList([FiLMedBlock(cfg.block_ch, n_extra) for _ in range(cfg.n_blocks)])

        self.film_gen = nn.Sequential(
            nn.Linear(question_dim, cfg.classifier_hidden),
            nn.ReLU(),
            nn.Linear(cfg.classifier_hidden, cfg.n_blocks * 2 * cfg.block_ch),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(cfg.block_ch, cfg.classifier_ch, 1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(cfg.classifier_ch, cfg.classifier_hidden),
            nn.ReLU(),
            nn.Linear(cfg.classifier_hidden, answer_dim),
        )

        if cfg.use_coords:
            lin = torch.linspace(-1.0, 1.0, self.encoder.spatial)
            yy, xx = torch.meshgrid(lin, lin, indexing='ij')
            self.register_buffer('coords', torch.stack([xx, yy])[None])
        else:
            self.coords = None

    def forward(self, batch: VQABatch, **overrides) -> VQAOutput:
        image = self.proj(self.encoder(batch['images']))
        question = self.question_encoder(batch['questions'])

        batch_size = image.size(0)
        params = self.film_gen(question).view(batch_size, self.n_blocks, 2, self.block_ch)

        coords = self.coords.expand(batch_size, -1, -1, -1) if self.coords is not None else None

        for i, block in enumerate(self.blocks):
            image = block(image, params[:, i, 0], params[:, i, 1], coords)

        return {'logits': self.classifier(image)}

    @classmethod
    def from_config(cls, model_cfg: VQAFiLMConfig, dataset: str, answer_dim: int,
        ) -> VQAFiLM:
        
        return cls(model_cfg, dataset, answer_dim)