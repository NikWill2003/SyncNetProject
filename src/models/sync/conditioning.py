"""The question policy: every pathway by which the question reaches the
computation, constructed in one place so the separation postulate is
checkable by enumeration.

The complete list -- there are no others anywhere in the sync models:
    1. encoder FiLM      f <- (1 + g(q)) * f + b(q)      perception, tinted once
    2. content FiLM      X <- LN((1 + g(q)) * X + b(q))  slot contents
    3. state inits       h_slots(q), h_head(q)           where reasoning starts
    4. the readout       MLP([h_head; q]) + Prior(q)     owned by readout.py

verify/verify_components.py asserts that zeroing pathways 1-3 and the
readout's q input leaves logits question-independent.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class QuestionPathways(nn.Module):

    def __init__(self, q_size: int, feat_ch: int, tok_dim: int,
                 n_modules: int, module_dim: int, head: bool = True):
        super().__init__()
        self.head = head
        self.grid_film_gamma = nn.Linear(q_size, feat_ch)
        self.grid_film_beta = nn.Linear(q_size, feat_ch)
        self.grid_norm = nn.GroupNorm(8, feat_ch, affine=True)
        self.film_gamma = nn.Linear(q_size, tok_dim)
        self.film_beta = nn.Linear(q_size, tok_dim)
        self.norm = nn.LayerNorm(tok_dim)
        self.h_init = nn.Sequential(nn.Linear(q_size, 64), nn.GELU(),
                                    nn.Linear(64, n_modules * module_dim))
        if head:
            self.head_init = nn.Sequential(nn.Linear(q_size, 64), nn.GELU(),
                                           nn.Linear(64, module_dim))
            self.head_embed = nn.Parameter(torch.randn(1, module_dim) / module_dim ** 0.5)
        self.n_modules, self.module_dim = n_modules, module_dim

    def encoder_film(self, f: Tensor, q: Tensor, pos_emb: Tensor) -> Tensor:
        f = f * (1 + self.grid_film_gamma(q))[..., None, None] + self.grid_film_beta(q)[..., None, None]
        return self.grid_norm(f) + pos_emb

    def content_film(self, X: Tensor, q: Tensor) -> Tensor:
        return self.norm(X * (1 + self.film_gamma(q)).unsqueeze(1) + self.film_beta(q).unsqueeze(1))

    def init_states(self, q_all: Tensor, questions: Tensor) -> tuple[Tensor, Tensor | None]:
        B = q_all.shape[0]
        h_slots = self.h_init(q_all).reshape(B, self.n_modules, self.module_dim)
        if not self.head:
            return h_slots, None
        h_head = self.head_init(questions) + self.head_embed.unsqueeze(0)
        return h_slots, h_head
