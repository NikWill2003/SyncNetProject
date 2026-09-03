"""BusNet -- the canonical SyncNet. One composition, readable top to bottom
against Chapter 3.

Perception binds by synchrony: an oscillator field settles until cells of
one object rotate together, and six exchangeable slots win their objects by
phase alignment alone (CompetitiveClaim -- content never enters the
competition). Communication is coherence: slots plus one question-holding
head are rows on a single shared wire; each writes m_i z_i^T, reads the sum
through its own orthonormal frame, echo cancelled (SharedBus). Phases are
driven Kuramoto oscillators whose stimulus each row computes from its own
state (KuramotoStep): what a module represents places it on the medium.
The answer is read from the head alone, plus a question prior
(HeadReadout): nothing reaches the output except what crossed the bus.

The template method splits the forward so the token control can subclass:
    forward = _bind(batch)  ->  _run(X, z_init, ...)
_bind consumes pixels; TokenBusNet overrides it to consume scenes instead.
A model consumes exactly one perceptual modality -- never both.

Config enums are the model's own experiment and nothing else:
    medium:    bus | lines | silent     (lines gated by cfg.gate)
    addresses: computed | static | open
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
from .components import (CompetitiveClaim, Exchangeable, KuramotoStep,
                         PrivateLines, SharedBus, SilentBus, phase_shuffle)
from .conditioning import QuestionPathways
from .field import FIELD_CH, FIELD_GROUPS, FIELD_OSC_D, OscillatorField, build_field_trunk
from .components.readout import HeadReadout

TOK_DIM = 64


@dataclass
class BusNetConfig(ModelConfig):
    # None | 'spatial' -- see CompetitiveClaim; 'spatial' adds a learned
    # per-slot cell bias to the competition logits (assembly aid).
    claim_prior: str | None = None
    # 'zero' learns the spatial prior from nothing (SoC winner); 'partition'
    # initialises it to a slot-per-region grid at claim_prior_scale, i.e. the
    # gated model's free binding as a starting point that competition can
    # refine -- the best-of-both candidate for SQOOP.
    claim_prior_init: str = 'zero'
    claim_prior_scale: float = 4.0
    readout_prior: bool = True
    # optional shared perception: {'name': 'cnn'|'patchify', ...} routes the
    # trunk through the common encoders (matched/ cells); None = FieldEncoder
    encoder: dict[str, Any] | None = None
    name: str = 'busnet'
    n_modules: int = 6
    module_dim: int = 96
    msg_dim: int = 4
    phase_dim: int = 6
    t_bus: int = 8
    dt: float = 0.5
    medium: str = 'bus'                  # bus | lines | silent
    gate: str = 'attn'                   # lines only: attn | full | phase
    addresses: str = 'computed'          # computed | static | open
    q_onehot_vocab: int | None = None    # None: raw float questions


def _dataset_spec(dataset: str):
    if dataset == 'sort_of_clevr':
        from ...datasets.soc import spec
    elif dataset == 'sqoop':
        from ...datasets.sqoop import spec
    else:
        raise ValueError(f'unknown dataset {dataset!r}')
    return spec


class BusNet(nn.Module):

    PHASE_OVERRIDES: ClassVar[tuple[str, ...]] = ('freeze', 'shuffle', 'zero')
    supported_callbacks: ClassVar[frozenset] = frozenset({'sync'})
    GATE_OVERRIDES: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: BusNetConfig, dataset: str, answer_dim: int):
        self._dataset = dataset
        super().__init__()
        self.cfg = cfg
        spec = _dataset_spec(dataset)
        self._dataset = dataset
        self.img_size = spec.IMG_SIZE
        q_raw = spec.QUESTION_SIZE
        self.q_size = q_raw * cfg.q_onehot_vocab if cfg.q_onehot_vocab else q_raw
        M, dm, d = cfg.n_modules, cfg.module_dim, cfg.phase_dim
        self.M, self.dm, self.d, self.T = M, dm, d, cfg.t_bus
        self.N = M + 1                                                       # slots + the head

        self._build_perception(cfg)
        self.pathways = QuestionPathways(self.q_size, FIELD_CH, TOK_DIM, M, dm)
        self.medium = self._build_medium(cfg)                                # first: the cell is sized by what it receives
        self.identity = self._build_identity(cfg)
        self.dynamics = KuramotoStep(self.N, dm, d, cfg.dt)
        self.readout = HeadReadout(dm, self.q_size, answer_dim, use_prior=cfg.readout_prior)
        if cfg.addresses == 'static':
            self.z_static = nn.Parameter(F.normalize(torch.randn(self.N, d), dim=-1))

    # -- construction, overridable by subclasses --------------------------
    def _build_perception(self, cfg: BusNetConfig) -> None:
        self.field_enc = build_field_trunk(cfg.encoder, self.img_size, self._dataset)
        S = self.field_enc.spatial
        self.pos_emb = nn.Parameter(0.02 * torch.randn(1, FIELD_CH, S, S))
        self.field = OscillatorField()
        self.binder = CompetitiveClaim(FIELD_CH, TOK_DIM, cfg.n_modules,
                                       FIELD_GROUPS, FIELD_OSC_D,
                                       per_module_anchors=self._per_module_anchors(cfg),
                                       claim_prior=getattr(cfg, "claim_prior", None),
                                       n_cells=self.field_enc.spatial ** 2,
                                       claim_prior_init=getattr(cfg, "claim_prior_init", "zero"),
                                       claim_prior_scale=getattr(cfg, "claim_prior_scale", 4.0))
        self.anchor_to_phase = nn.Linear(FIELD_GROUPS * FIELD_OSC_D, cfg.phase_dim)

    def _per_module_anchors(self, cfg: BusNetConfig) -> bool:
        return False

    def _build_identity(self, cfg: BusNetConfig):
        return Exchangeable(self.N, TOK_DIM + self.medium.out_dim,
                            cfg.module_dim, cfg.n_modules)

    def _build_medium(self, cfg: BusNetConfig):
        if cfg.medium == 'bus':
            return SharedBus(cfg.module_dim, cfg.msg_dim, cfg.phase_dim)
        if cfg.medium == 'silent':
            return SilentBus(cfg.module_dim, cfg.msg_dim, cfg.phase_dim)
        if cfg.medium == 'lines':
            return PrivateLines(cfg.module_dim, cfg.msg_dim, cfg.phase_dim,
                                self.N, gate=cfg.gate)
        raise ValueError(f'unknown medium {cfg.medium!r}')

    # -- the question -----------------------------------------------------
    def _encode_q(self, questions: Tensor) -> Tensor:
        if self.cfg.q_onehot_vocab:
            return F.one_hot(questions.long(), self.cfg.q_onehot_vocab).float().flatten(-2)
        return questions.float()

    # -- bind: pixels -> slot contents + starting addresses ---------------
    def _bind(self, batch: VQABatch, q_all: Tensor,
              prior_perm: Tensor | None = None) -> tuple[Tensor, Tensor | None, dict]:
        f = self.field_enc(batch['images'])
        f = self.pathways.encoder_film(f, q_all, self.pos_emb)
        zf = self.field(f)
        Zt = self.field.to_tokens(zf)                                        # (B, P, K, d_f)
        feats = f.flatten(2).transpose(1, 2)                                 # (B, P, fch)
        X, anchors, reads = self.binder(feats, Zt, prior_perm)
        z_slots = F.normalize(self.anchor_to_phase(anchors.flatten(2)), dim=-1)
        with torch.no_grad():                                                # detector input, not a compute path
            field_R = Zt.mean(1).norm(dim=-1).mean()
        return X, z_slots, {'slot_reads': reads, 'anchors': anchors,
                            'field_R': field_R}

    # -- run: the bus loop and the readout --------------------------------
    def _run(self, X: Tensor, z_slots: Tensor | None, q_all: Tensor,
             questions: Tensor, phase_override: str | None) -> tuple[Tensor, Tensor]:
        B = X.shape[0]
        h_slots, h_head = self.pathways.init_states(q_all, questions)
        h = torch.cat([self.identity.embed(h_slots), h_head], 1)
        X = torch.cat([X, torch.zeros(B, 1, X.shape[-1], device=X.device, dtype=X.dtype)], 1)

        z = F.normalize(torch.randn(B, self.N, self.d, device=X.device, dtype=X.dtype), dim=-1)
        if z_slots is not None:
            z = torch.cat([z_slots, z[:, self.M:]], 1)
        if self.cfg.addresses == 'static':
            z = F.normalize(self.z_static, dim=-1)[None].expand(B, -1, -1).to(X.dtype)
        elif self.cfg.addresses == 'open' or phase_override == 'zero':
            e1 = torch.zeros(self.d, device=X.device, dtype=X.dtype)
            e1[0] = 1.0
            z = e1.expand(B, self.N, self.d).clone()
        if phase_override == 'shuffle':
            z = phase_shuffle(z, self.M)

        evolve = (self.cfg.addresses == 'computed' and phase_override not in ('freeze', 'zero'))
        for _ in range(self.T):
            r = self.medium(h, z)
            inp = torch.cat([X, r], -1)
            h = self.identity.step(inp, h)
            if evolve:
                z = self.dynamics(z, h)
        return h, z

    # -- override hooks: identity in the base, real in subclasses ---------
    def _prior_perm(self, phase_override: str | None, batch: VQABatch):
        return None                                                          # exchangeable: nothing to permute

    def _loop_override(self, phase_override: str | None) -> str | None:
        return phase_override

    # -- forward ----------------------------------------------------------
    def forward(self, batch: VQABatch, phase_override: str | None = None,
                **_: Any) -> VQAOutput:
        questions = self._encode_q(batch['questions'])
        squeeze = questions.dim() == 2
        if squeeze:
            questions = questions.unsqueeze(1)                               # (B, 1, q)
        q_all = questions.reshape(questions.shape[0], -1)

        X, z_slots, extras = self._bind(batch, q_all,
                                        prior_perm=self._prior_perm(phase_override, batch))
        X = self.pathways.content_film(X, q_all)
        h, z = self._run(X, z_slots, q_all, questions,
                         self._loop_override(phase_override))

        logits = self.readout(h[:, self.M:], questions)
        if squeeze:
            logits = logits.squeeze(1)
        return {'logits': logits, 'traces': {'phases': z, **extras}}

    @classmethod
    def from_config(cls, model_cfg: BusNetConfig, dataset: str, answer_dim: int):
        return cls(model_cfg, dataset, answer_dim)
