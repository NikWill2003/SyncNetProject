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
    """Answer from the head module's final state. `use_prior=False` removes
    the question-only prior term: the head is initialised from the question
    anyway, so the term adds no expressivity, only a path to the answer that
    bypasses the medium."""

    def __init__(self, module_dim: int, q_size: int, answer_dim: int, hidden: int = 128,
                 use_prior: bool = True):
        super().__init__()
        self.head_out = nn.Sequential(nn.Linear(module_dim + q_size, hidden), nn.GELU(),
                                      nn.Linear(hidden, answer_dim))
        self.prior_head = (nn.Sequential(nn.Linear(q_size, hidden), nn.GELU(),
                                         nn.Linear(hidden, answer_dim))
                           if use_prior else None)

    def forward(self, h_head: Tensor, questions: Tensor) -> Tensor:
        out = self.head_out(torch.cat([h_head, questions], -1))
        return out + self.prior_head(questions) if self.prior_head is not None else out


class PooledReadout(nn.Module):
    """Integrating readout for module-set models: a question-conditioned
    attention over module states, then one MLP. Unlike VoteReadout (a sum of
    independent per-module votes) it CAN combine modules at readout, so the
    vote/pooled contrast asks whether forcing all integration through the
    medium is what makes modules differentiate."""

    def __init__(self, module_dim: int, q_size: int, answer_dim: int, hidden: int = 128,
                 use_prior: bool = True):
        super().__init__()
        self.q_key = nn.Linear(q_size, module_dim)
        self.out = nn.Sequential(nn.Linear(module_dim + q_size, hidden), nn.GELU(),
                                 nn.Linear(hidden, answer_dim))
        self.prior_head = (nn.Sequential(nn.Linear(q_size, hidden), nn.GELU(),
                                         nn.Linear(hidden, answer_dim))
                           if use_prior else None)

    def forward(self, h_modules: Tensor, questions: Tensor) -> Tensor:
        k = self.q_key(questions).unsqueeze(1)                       # (B,1,dm)
        att = torch.softmax((h_modules * k).sum(-1) / h_modules.shape[-1] ** 0.5, dim=1)
        pooled = (att.unsqueeze(-1) * h_modules).sum(1)                # (B,dm)
        out = self.out(torch.cat([pooled, questions], -1))
        return out + self.prior_head(questions) if self.prior_head is not None else out


class VoteReadout(nn.Module):

    def __init__(self, module_dim: int, q_size: int, answer_dim: int, hidden: int = 128,
                 use_prior: bool = True):
        super().__init__()
        self.vote = nn.Sequential(nn.Linear(module_dim + q_size, hidden), nn.GELU(),
                                  nn.Linear(hidden, answer_dim))
        self.prior_head = (nn.Sequential(nn.Linear(q_size, hidden), nn.GELU(),
                                         nn.Linear(hidden, answer_dim))
                           if use_prior else None)

    def forward(self, h_modules: Tensor, questions: Tensor) -> Tensor:
        B, N, dm = h_modules.shape
        q = questions.unsqueeze(1).expand(B, N, questions.shape[-1])
        votes = self.vote(torch.cat([h_modules, q], -1)).sum(1)
        return votes + self.prior_head(questions) if self.prior_head is not None else votes
