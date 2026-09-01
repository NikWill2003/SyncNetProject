"""TokenBusNet -- the mechanism isolator. Ground-truth scene descriptors in
place of pixels: same rows, same bus, same dynamics, same head, perception
assumed away. It is what localises a pixel failure to perception (the
identical medium goes chance -> perfect in tens of steps when fed clean
tokens on the task the pixel model cannot leave the floor of).

SEGREGATION, enforced by construction: this model builds no encoder, no
field, no pixel binder -- its forward consumes batch['scenes'] and cannot
be handed pixels; the pixel models' _bind consumes images and cannot be
handed scenes. A model consumes exactly one perceptual modality.
verify/verify_segregation.py asserts both directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .busnet import TOK_DIM, BusNet, BusNetConfig
from .components import GivenTokens
from ...core.contracts import VQABatch

# per-dataset scene schemas -> token input width
#   sqoop          scenes (B, 5, 3) = letter idx, x, y   -> onehot(36) + xy   = 38
#   sort_of_clevr  scenes (B, 6, 3) = x, y, shape bit    -> onehot(6 colours,
#                  row index) + shape + xy                                 = 9
TOKEN_SPECS = {'sqoop': 36 + 2, 'sort_of_clevr': 6 + 1 + 2}


@dataclass
class TokenBusNetConfig(BusNetConfig):
    name: str = 'token_busnet'
    n_modules: int = 5                   # one row per scene object


class TokenBusNet(BusNet):

    supported_callbacks = frozenset({'sync'})

    def _build_perception(self, cfg: TokenBusNetConfig) -> None:
        self.binder = GivenTokens(TOKEN_SPECS[self._dataset], TOK_DIM)

    def _bind(self, batch: VQABatch, q_all: Tensor,
              prior_perm: Tensor | None = None):
        scenes = batch['scenes']
        B, M = scenes.shape[:2]
        if self._dataset == 'sqoop':                                         # (B, 5, 3): letter, x, y
            ident = F.one_hot(scenes[..., 0].long(), 36).float()
            rest = scenes[..., 1:].float() / max(self.img_size - 1, 1)
        else:                                                                # (B, 6, 3): x, y, shape; row = colour
            colour = torch.eye(6, device=scenes.device)[None].expand(B, M, 6).float()
            shape = scenes[..., 2:3].float()
            xy = scenes[..., :2].float() / max(self.img_size - 1, 1)
            ident, rest = colour, torch.cat([shape, xy], -1)
        X = self.binder(torch.cat([ident, rest], -1))
        return X, None, {}                                                   # no anchors: addresses start random
