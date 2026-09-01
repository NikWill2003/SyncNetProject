from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from ...core.config import ModelConfig
from ...core.contracts import VQABatch, VQAOutput
from ..common.img_enc import build_image_encoder
from ..common.qst_enc import build_question_encoder


@dataclass
class VQAConvConfig(ModelConfig):
    name: str = 'conv'
    q_encoder: dict[str, Any] = field(default_factory=lambda: {'name': 'identity'})

    hidden_dim: int = 256
    encoder: dict[str, Any] = field(default_factory=lambda: {
        'name': 'cnn', 'ch': 128,
    })


class VQAConv(nn.Module):

    allowed_q_encoders = {'identity', 'mlp', 'lstm'}

    def __init__(self, cfg: VQAConvConfig, dataset: str, answer_dim: int) -> None:
        super().__init__()

        self.question_encoder = build_question_encoder(cfg.q_encoder, dataset, self.allowed_q_encoders)
        self.encoder = build_image_encoder(cfg.encoder, dataset)

        question_shape = self.question_encoder.output_shape
        if len(question_shape) != 1:
            raise ValueError(f'VQAConv requires a vector question representation, got {question_shape}')

        question_dim = question_shape[0]
        image_dim = self.encoder.ch * self.encoder.spatial * self.encoder.spatial

        self.head = nn.Sequential(
            nn.Linear(image_dim + question_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, answer_dim),
        )

    def forward(self, batch: VQABatch, **overrides) -> VQAOutput:

        image = self.encoder(batch['images']).flatten(1)
        question = self.question_encoder(batch['questions'])
        return {'logits': self.head(torch.cat([image, question], dim=1))}

    @classmethod
    def from_config(cls, model_cfg: VQAConvConfig, dataset: str, answer_dim: int
        ) -> VQAConv:

        return cls(model_cfg, dataset, answer_dim)

