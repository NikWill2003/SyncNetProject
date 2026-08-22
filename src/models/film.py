"""FiLM (Perez et al. 2018), reported as a baseline on both tasks used here.

Reference architecture: `arXiv:1709.07871`; the authors' implementation is
`github.com/ethanjperez/film`, where the network is `vr/models/filmed_net.py`
and the conditioning generator is `vr/models/film_gen.py`. Written here
from the published description against this codebase's shared-model
interface rather than adapted from that source.

Two deliberate departures, both forced by the setting.

THE GENERATOR IS REPLACED, NOT PORTED. Theirs is a recurrent network over
tokenised natural language, producing per-block modulation parameters.
Neither task here uses natural language: Sort-of-CLEVR questions are
fixed-width binary vectors and SQOOP questions are three token indices.
The generator is therefore an MLP over this codebase's `QuestionEncoder`
output, emitting the same `n_blocks x 2 x ch` parameters.

THE PRETRAINED STEM IS DROPPED. The reference pipeline consumes ResNet
features from a CLEVR preprocessing stage. At 64--75 px on synthetic
scenes a trainable stem is the appropriate choice, so the shared
`CNNEncoder` supplies the features and the FiLMed residual blocks sit on
top of it.

NOTE ON NAMING. This is not the same thing as the `q_conditioning='film'`
axis of the transformer and syncnet. That axis applies ONE modulation to
the encoder output before tokenisation; this model applies a SEPARATE
modulation inside every residual block, with parameters predicted per
block. They should not share a label in any results table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..core.config import ModelConfig
from .question_encoders import QuestionEncoder


@dataclass
class VQAFiLMConfig(ModelConfig):
    n_blocks: int = 4
    block_ch: int = 128
    classifier_ch: int = 512
    classifier_hidden: int = 1024
    use_coords: bool = True     # coordinate maps into each block

    encoder: dict[str, Any] = field(default_factory=lambda: {
        'name': 'cnn', 'ch': 128, 'hidden': 64,
    })


class FiLMedBlock(nn.Module):
    """One residual block whose normalised activations are modulated by
    (gamma, beta) predicted from the question.

    The FiLM operation itself is a feature-wise affine transform,
    `gamma * x + beta`, broadcast over the spatial dimensions. The
    normalisation immediately before it is affine-free, since the
    generator supplies the scale and shift.
    """

    def __init__(self, ch: int, n_extra: int = 0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch + n_extra, ch, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm = nn.BatchNorm2d(ch, affine=False)

    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor,
                extra: Tensor | None = None) -> Tensor:
        inp = x if extra is None else torch.cat([x, extra], dim=1)
        h = torch.relu(self.conv1(inp))
        residual = h
        h = self.norm(self.conv2(h))
        h = h * (1.0 + gamma[..., None, None]) + beta[..., None, None]
        return torch.relu(h) + residual


class VQAFiLM(nn.Module):

    has_rotors = False
    is_syncnet = False

    def __init__(
            self,
            q_encoder: QuestionEncoder,
            encoder: nn.Module,
            answer_dim: int,
            n_blocks: int = 4,
            block_ch: int = 128,
            classifier_ch: int = 512,
            classifier_hidden: int = 1024,
            use_coords: bool = True,
            ) -> None:
        super().__init__()
        self.q_encoder = q_encoder
        self.encoder = encoder
        self.n_blocks = n_blocks
        self.block_ch = block_ch

        self.proj = (
            nn.Conv2d(encoder.ch, block_ch, 1)
            if encoder.ch != block_ch else nn.Identity()
        )

        n_extra = 2 if use_coords else 0
        self.blocks = nn.ModuleList(
            FiLMedBlock(block_ch, n_extra) for _ in range(n_blocks)
        )

        # the generator: question -> per-block (gamma, beta)
        self.film_gen = nn.Sequential(
            nn.Linear(q_encoder.out_dim, classifier_hidden), nn.ReLU(),
            nn.Linear(classifier_hidden, n_blocks * 2 * block_ch),
        )

        s = encoder.spatial
        self.classifier = nn.Sequential(
            nn.Conv2d(block_ch, classifier_ch, 1), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            nn.Linear(classifier_ch, classifier_hidden), nn.ReLU(),
            nn.Linear(classifier_hidden, answer_dim),
        )

        if use_coords:
            lin = torch.linspace(-1.0, 1.0, s)
            yy, xx = torch.meshgrid(lin, lin, indexing='ij')
            self.register_buffer('coords', torch.stack([xx, yy])[None])
        else:
            self.coords = None

    def forward(self, images: Tensor, questions: Tensor, **kwargs) -> Tensor:
        B = images.size(0)
        x = self.proj(self.encoder(images))                  # (B, ch, S, S)

        params = self.film_gen(self.q_encoder.flat(questions))
        params = params.view(B, self.n_blocks, 2, self.block_ch)

        extra = None
        if self.coords is not None:
            extra = self.coords.expand(B, -1, -1, -1)

        for i, block in enumerate(self.blocks):
            x = block(x, params[:, i, 0], params[:, i, 1], extra)

        return self.classifier(x)
