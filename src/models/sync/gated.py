"""GatedNet -- M1, the gated lineage recomposed from the component library.

Binding by fiat (PartitionRead: a fixed spatial grid, one module per
region), identity by parameters (PhaseNative: private cells + embeddings --
the modules are individuals), communication on PRIVATE LINES with a gate
deciding who is heard, phases on the scalar circle (the same KuramotoStep
at d = 2), votes summed at the readout. This is the architecture whose
collapse motivated the thesis: wherever a content gate can make the routing
decision directly, optimisation abandons the phase (L2) -- gate='attn' is
that competitor, gate='phase' the synchrony arm, gate='full' the bandwidth
ceiling. Rebuilt here so the era's claims rerun under the frozen protocol
on the same parts as the canonical model, not as an incomparable ancestor.

No head row: every module votes. phase_override='zero' sets all phases
equal, which for the phase gate IS the open gate -- the two suites meet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core.config import ModelConfig
from ...core.contracts import VQABatch, VQAOutput
from .busnet import TOK_DIM, _dataset_spec
from .components import (KuramotoStep, PartitionRead, PhaseNative,
                         PrivateLines, phase_shuffle)
from .components.readout import PooledReadout, VoteReadout
from .conditioning import QuestionPathways
from .field import FIELD_CH, build_field_trunk


@dataclass
class GatedNetConfig(ModelConfig):
    encoder: dict[str, Any] | None = None
    readout: str = 'vote'          # 'vote' (sum of per-module votes) | 'pooled'
    readout_prior: bool = True     # keep the question-only prior term
    name: str = 'gated'
    grid: int = 2                        # grid^2 modules, one per region
    module_dim: int = 96
    msg_dim: int = 16
    phase_dim: int = 2                   # the scalar circle
    t_steps: int = 8
    dt: float = 0.5
    gate: str = 'phase'                  # phase | attn | full
    q_onehot_vocab: int | None = None


class GatedNet(nn.Module):

    PHASE_OVERRIDES: ClassVar[tuple[str, ...]] = ('freeze', 'shuffle', 'zero')
    supported_callbacks: ClassVar[frozenset] = frozenset({'sync'})
    GATE_OVERRIDES: ClassVar[tuple[str, ...]] = ('open', 'zero')

    def __init__(self, cfg: GatedNetConfig, dataset: str, answer_dim: int):
        self._dataset = dataset
        super().__init__()
        self.cfg = cfg
        spec = _dataset_spec(dataset)
        self.img_size = spec.IMG_SIZE
        q_raw = spec.QUESTION_SIZE
        self.q_size = q_raw * cfg.q_onehot_vocab if cfg.q_onehot_vocab else q_raw
        M, dm, d = cfg.grid ** 2, cfg.module_dim, cfg.phase_dim
        self.M, self.dm, self.d, self.T = M, dm, d, cfg.t_steps
        self.N = M                                                           # no head: votes

        self.field_enc = build_field_trunk(cfg.encoder, self.img_size, self._dataset)
        S = self.field_enc.spatial
        self.pos_emb = nn.Parameter(0.02 * torch.randn(1, FIELD_CH, S, S))
        self.binder = PartitionRead(FIELD_CH, TOK_DIM, grid=cfg.grid)
        self.pathways = QuestionPathways(self.q_size, FIELD_CH, TOK_DIM, M, dm, head=False)
        self.identity = PhaseNative(self.N, TOK_DIM + self._r_dim(cfg), dm, M,
                                    private_cells=True, per_module_anchors=False)
        self.medium = PrivateLines(dm, cfg.msg_dim, d, self.N, gate=cfg.gate)
        self.dynamics = KuramotoStep(self.N, dm, d, cfg.dt)
        Readout = {'vote': VoteReadout, 'pooled': PooledReadout}[cfg.readout]
        self.readout = Readout(dm, self.q_size, answer_dim, use_prior=cfg.readout_prior)

    def _r_dim(self, cfg: GatedNetConfig) -> int:
        return cfg.msg_dim * cfg.grid ** 2 if cfg.gate == 'full' else cfg.msg_dim

    def _encode_q(self, questions: Tensor) -> Tensor:
        if self.cfg.q_onehot_vocab:
            return F.one_hot(questions.long(), self.cfg.q_onehot_vocab).float().flatten(-2)
        return questions.float()

    def forward(self, batch: VQABatch, phase_override: str | None = None,
                t_override: int | None = None,
                gate_override: str | None = None, **_: Any) -> VQAOutput:
        questions = self._encode_q(batch['questions'])
        squeeze = questions.dim() == 2
        if squeeze:
            questions = questions.unsqueeze(1)
        q_all = questions.reshape(questions.shape[0], -1)
        B = q_all.shape[0]

        f = self.field_enc(batch['images'])
        f = self.pathways.encoder_film(f, q_all, self.pos_emb)
        X = self.binder(f)                                                   # (B, M, tok): binding by fiat
        X = self.pathways.content_film(X, q_all)

        h, _ = self.pathways.init_states(q_all, questions)
        h = self.identity.embed(h)

        z = F.normalize(torch.randn(B, self.N, self.d, device=X.device, dtype=X.dtype), dim=-1)
        if phase_override == 'zero':
            e1 = torch.zeros(self.d, device=X.device, dtype=X.dtype)
            e1[0] = 1.0
            z = e1.expand(B, self.N, self.d).clone()
        if phase_override == 'shuffle':
            z = phase_shuffle(z, self.M)

        evolve = phase_override not in ('freeze', 'zero')
        for _t in range(self.T if t_override is None else int(t_override)):
            r = self.medium(h, z, gate_override=gate_override)
            h = self.identity.step(torch.cat([X, r], -1), h)
            if evolve:
                z = self.dynamics(z, h)

        logits = self.readout(h, questions.squeeze(1) if not squeeze else questions[:, 0])
        return {'logits': logits, 'traces': {'phases': z}}

    @classmethod
    def from_config(cls, model_cfg: GatedNetConfig, dataset: str, answer_dim: int):
        return cls(model_cfg, dataset, answer_dim)
