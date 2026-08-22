"""Relation Network (Santoro et al. 2017), the model Sort-of-CLEVR was
introduced with.

Reference architecture: `arXiv:1706.01427`; the widely used PyTorch
implementation is `github.com/kimhc6028/relational-networks`, which is
also where this project's ternary question family is derived from.
Written here from the published description against this codebase's
shared-model interface rather than adapted from that source, so the axes,
encoders and question encoders match every other model in `src/models/`.

The architecture is
    RN(O) = f_phi( sum_{i,j} g_theta(o_i, o_j, q) )
where the objects o are the cells of a convolutional feature map, each
tagged with its own normalised coordinates.

Two properties matter for how it is reported.

COORDINATE TAGGING IS NOT OPTIONAL. A convolutional feature map is
translation-equivariant, so a cell carries what is there but not where.
Every relation in both tasks is a position predicate, so without the
appended (x, y) the pairs are unanswerable in principle. This is done
here rather than in `encoders.py` so it cannot silently alter the other
models that share those encoders.

CONDITIONING IS INTRINSIC. The question joins *each pair* inside
g_theta, so unlike the transformer and syncnet this model cannot take the
`q_conditioning` axis. Any comparison against it is therefore not matched
on that axis, and should say so.
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
class VQARelNetConfig(ModelConfig):
    g_hidden: int = 256
    g_layers: int = 4
    f_hidden: int = 256
    f_dropout: float = 0.0

    # Objects are the cells of the feature map, so the number of pairs is
    # the square of the cell count and dominates the cost: a 5x5 grid is
    # 625 pairs, a 10x10 grid is 10,000. The map is adaptively pooled to
    # this side length before pairing, which keeps the cost fixed as the
    # image size or encoder changes.
    pair_spatial: int = 5

    encoder: dict[str, Any] = field(default_factory=lambda: {
        'name': 'cnn', 'ch': 128, 'hidden': 64,
    })


class VQARelNet(nn.Module):

    has_rotors = False
    is_syncnet = False

    def __init__(
            self,
            q_encoder: QuestionEncoder,
            encoder: nn.Module,
            answer_dim: int,
            g_hidden: int = 256,
            g_layers: int = 4,
            f_hidden: int = 256,
            f_dropout: float = 0.0,
            pair_spatial: int = 5,
            ) -> None:
        super().__init__()
        self.q_encoder = q_encoder
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(pair_spatial)
        self.n_obj = pair_spatial * pair_spatial

        obj_dim = encoder.ch + 2                     # + normalised (x, y)
        g_in = 2 * obj_dim + q_encoder.out_dim

        layers: list[nn.Module] = []
        d = g_in
        for _ in range(g_layers):
            layers += [nn.Linear(d, g_hidden), nn.ReLU()]
            d = g_hidden
        self.g_theta = nn.Sequential(*layers)

        self.f_phi = nn.Sequential(
            nn.Linear(g_hidden, f_hidden), nn.ReLU(),
            nn.Dropout(f_dropout),
            nn.Linear(f_hidden, f_hidden), nn.ReLU(),
            nn.Linear(f_hidden, answer_dim),
        )

        # (2, S, S) coordinate maps in [-1, 1], registered so they move
        # with .to(device) and are saved with the model.
        lin = torch.linspace(-1.0, 1.0, pair_spatial)
        yy, xx = torch.meshgrid(lin, lin, indexing='ij')
        self.register_buffer('coords', torch.stack([xx, yy])[None])

    def forward(self, images: Tensor, questions: Tensor, **kwargs) -> Tensor:
        B = images.size(0)
        feats = self.pool(self.encoder(images))              # (B, ch, S, S)
        feats = torch.cat([feats, self.coords.expand(B, -1, -1, -1)], dim=1)
        obj = feats.flatten(2).transpose(1, 2)               # (B, N, ch+2)

        q = self.q_encoder.flat(questions)                   # (B, |q|)
        N = obj.size(1)
        left = obj.unsqueeze(2).expand(-1, -1, N, -1)
        right = obj.unsqueeze(1).expand(-1, N, -1, -1)
        qq = q.unsqueeze(1).unsqueeze(1).expand(-1, N, N, -1)

        pairs = torch.cat([left, right, qq], dim=-1)         # (B, N, N, D)
        rel = self.g_theta(pairs.flatten(1, 2))              # (B*N*N, G)
        rel = rel.view(B, N * N, -1).sum(dim=1)              # (B, G)
        return self.f_phi(rel)
