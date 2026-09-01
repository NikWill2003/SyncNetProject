"""O -- how the answer leaves the system.

HeadReadout (canonical): the head row holds only the question and what
crossed the medium, so the silent medium is the question-only floor by
construction and every point above it is attributable to communication.
VoteReadout (gated): per-module vote MLPs, summed -- the integration-free
alternative, needed for the gated model's faithful recomposition.
Both carry the prior term; the question reaches the logits through nothing
else.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class HeadReadout(nn.Module):

    def __init__(self, module_dim: int, q_size: int, answer_dim: int, hidden: int = 128):
        super().__init__()
        self.head_out = nn.Sequential(nn.Linear(module_dim + q_size, hidden), nn.GELU(),
                                      nn.Linear(hidden, answer_dim))
        self.prior_head = nn.Sequential(nn.Linear(q_size, hidden), nn.GELU(),
                                        nn.Linear(hidden, answer_dim))

    def forward(self, h_head: Tensor, questions: Tensor) -> Tensor:
        return self.head_out(torch.cat([h_head, questions], -1)) + self.prior_head(questions)


class VoteReadout(nn.Module):

    def __init__(self, module_dim: int, q_size: int, answer_dim: int, hidden: int = 128):
        super().__init__()
        self.vote = nn.Sequential(nn.Linear(module_dim + q_size, hidden), nn.GELU(),
                                  nn.Linear(hidden, answer_dim))
        self.prior_head = nn.Sequential(nn.Linear(q_size, hidden), nn.GELU(),
                                        nn.Linear(hidden, answer_dim))

    def forward(self, h_modules: Tensor, questions: Tensor) -> Tensor:
        B, N, dm = h_modules.shape
        q = questions.unsqueeze(1).expand(B, N, questions.shape[-1])
        votes = self.vote(torch.cat([h_modules, q], -1)).sum(1)
        return votes + self.prior_head(questions)
