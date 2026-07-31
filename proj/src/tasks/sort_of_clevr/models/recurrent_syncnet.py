from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING

from ....core.config import ModelConfig
from ....core.encoders import (
    PatchifyEncoder, CNNEncoder, EncoderConfig
)
from ..config import SortOfClevrDataConfig
from ..contracts import SortOfClevrOutput
from ..data import constants as C

Encoder = PatchifyEncoder | CNNEncoder

"""RecurrentSyncNet: phase as a *module state*, not a content property.

Design (from notebook, 2026-07):
--------------------------------
The recurring conceptual problem with rotor-per-feature models (V1) is
"what does the phase of a feature vector mean?". Here phase is instead a
single scalar state theta_i carried by each recurrent module i, alongside
its hidden state h_i. Phase does exactly one job: it *gates communication*.

Per step t (order matters -- content first, then phase, as in the notes
where the coupling C reads h^{(t+1)}):

  1. gates       g_ij = 0.5 * (1 + cos(theta_i - theta_j))       in [0, 1]
  2. messages    msg_i = (1/M) sum_j g_ij * W_msg h_j            phase-gated
  3. input       a_i   = softmax_p(beta * cos(W_q h_i, c_p)) c   attention
  4. content     h_i  <- GRUCell([a_i ; msg_i], h_i)             S(h, A (x) x)
  5. coupling    C_ij = sin(theta_j - theta_i) * k([h_i ; h_j])
     phase       theta_i <- theta_i + dt * (omega_i + sum_j K_ij C_ij)

with k: R^{2d} -> [-1, 1] an MLP (tanh head), so the *content decides*
whether two modules attract (join a coalition) or repel (stay apart);
K is a learnable M x M coupling-strength matrix (the k_ij in the notes).

Sign convention: the notes write sin(theta_i - theta_j); standard Kuramoto
attraction under positive coupling is sin(theta_j - theta_i). We use the
latter so k > 0 means "synchronise / open the gate". Since k is free-signed
the two conventions are equivalent up to a sign flip the network can learn,
but this way the init story is interpretable.

Omega in the notes (theta <- Omega theta + ...) is realised as a learnable
per-module natural frequency omega_i entering additively under the Euler
step -- i.e. rotation at rate omega_i -- which is the standard
discretisation and keeps theta on the circle (wrapped implicitly by
cos/sin; theta itself is unconstrained).

Readout: per-module content read c_i = A_i^T b_content (as in V3), concat
over modules with the question, MLP head.

Interface notes:
  * t_override: run the dynamics for a different number of steps at test
    time (T-generalisation / T-variance ablations).
  * phase init is random uniform on [0, 2pi) per forward pass -- so
    repeated forwards give the variance estimate the T-variance ablation
    needs. Set deterministic_phase=True to zero-init instead.
  * scramble_state: spatially permutes the encoded tokens the modules
    read, matching the causal intervention in sync_metrics.
  * out['metrics'] reports phase_R (order parameter over module phases),
    gate_mean and gate_offdiag (how open the communication graph is), so
    every logged step tracks whether coalitions actually form.
"""


@dataclass
class SortOfClevrRecurrentSyncNetConfig(ModelConfig):
    name: str = 'sort_of_clevr_recurrent_syncnet'
    forward_args: dict[str, Any] = field(default_factory=dict)

    # modules
    n_modules: int = 4
    module_dim: int = 128        # h_i dimension
    msg_dim: int = 64            # W_msg h_j dimension

    # input attention
    use_film: bool = True
    beta_init: float = 5.0
    learn_beta: bool = True
    content_dim: int = 8         # per-module content read for the head
    query_hidden: int = 64       # question -> h_i init MLP width

    # phase dynamics
    T: int = 6
    dt: float = 0.1
    omega_init: float = 0.5      # spread of per-module natural frequencies
    k_hidden: int = 64           # coupling MLP width
    deterministic_phase: bool = False

    # readout
    hidden_dim: int = 128

    encoder_cfg: EncoderConfig = MISSING


class SortOfClevrRecurrentSyncNet(nn.Module):

    has_rotors = False
    is_syncnet = False  # spatial-state sync callbacks don't apply; the
                        # model reports its own phase metrics instead

    def __init__(
            self,
            encoder: Encoder,
            answer_dim: int,
            q_dim: int,
            n_modules: int = 4,
            module_dim: int = 128,
            msg_dim: int = 64,
            use_film: bool = True,
            beta_init: float = 5.0,
            learn_beta: bool = True,
            content_dim: int = 8,
            query_hidden: int = 64,
            T: int = 6,
            dt: float = 0.1,
            omega_init: float = 0.5,
            k_hidden: int = 64,
            deterministic_phase: bool = False,
            hidden_dim: int = 128,
            ):
        super().__init__()

        ch = encoder.ch

        self.encoder = encoder
        self.spatial = encoder.spatial
        self.n_tokens = encoder.n_tokens
        self.ch = ch
        self.q_dim = q_dim
        self.answer_dim = answer_dim

        self.M = n_modules
        self.d = module_dim
        self.msg_dim = msg_dim
        self.use_film = use_film
        self.content_dim = content_dim
        self.T = T
        self.dt = dt
        self.deterministic_phase = deterministic_phase

        if use_film:
            self.film_gamma = nn.Linear(q_dim, ch)
            self.film_beta = nn.Linear(q_dim, ch)
        self.c_norm = nn.GroupNorm(8, ch, affine=True)

        # question -> per-module initial hidden state
        self.h_init = nn.Sequential(
            nn.Linear(q_dim, query_hidden),
            nn.GELU(),
            nn.Linear(query_hidden, n_modules * module_dim),
        )

        # input attention: A_i^{(t)} from h_i vs tokens
        self.attn_query = nn.Linear(module_dim, ch)
        log_beta = torch.tensor(float(np.log(beta_init)))
        if learn_beta:
            self.log_beta = nn.Parameter(log_beta)
        else:
            self.register_buffer('log_beta', log_beta)

        # phase-gated messaging
        self.msg_proj = nn.Linear(module_dim, msg_dim)

        # content update S: GRU on [attended input ; messages]
        self.cell = nn.GRUCell(ch + msg_dim, module_dim)

        # phase machinery
        self.omega = nn.Parameter(
            omega_init * torch.linspace(-1.0, 1.0, n_modules)
        )
        self.K = nn.Parameter(torch.ones(n_modules, n_modules))
        self.k_mlp = nn.Sequential(
            nn.Linear(2 * module_dim, k_hidden),
            nn.GELU(),
            nn.Linear(k_hidden, 1),
            nn.Tanh(),
        )

        # readout
        self.content_head = nn.Linear(ch, content_dim)
        self.head = nn.Sequential(
            nn.Linear(n_modules * content_dim + q_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, answer_dim),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _gates(theta: torch.Tensor) -> torch.Tensor:
        # theta: (B, M) -> g: (B, M, M), g_ij = (1 + cos(ti - tj)) / 2
        diff = theta.unsqueeze(-1) - theta.unsqueeze(-2)
        return 0.5 * (1.0 + torch.cos(diff))

    def _phase_step(
            self, theta: torch.Tensor, h: torch.Tensor
            ) -> torch.Tensor:
        # theta: (B, M), h: (B, M, d)
        B, M, d = h.shape

        hi = h.unsqueeze(2).expand(B, M, M, d)
        hj = h.unsqueeze(1).expand(B, M, M, d)
        k = self.k_mlp(torch.cat([hi, hj], dim=-1)).squeeze(-1)  # (B, M, M)

        diff = theta.unsqueeze(1) - theta.unsqueeze(2)  # (B, M, M): tj - ti
        coupling = (self.K * torch.sin(diff) * k).sum(dim=-1)   # (B, M)

        return theta + self.dt * (self.omega + coupling)

    # ------------------------------------------------------------------

    def forward(
            self,
            images: torch.Tensor,
            questions: torch.Tensor,
            t_override: Optional[int] = None,
            return_trace: bool = False,
            scramble_state: bool = False,
            **kwargs,
            ) -> SortOfClevrOutput:

        q = questions.float()
        B = images.shape[0]
        M, d = self.M, self.d
        T = t_override if t_override is not None else self.T
        device = images.device

        c = self.encoder(images)                       # (B, ch, H, W)
        if self.use_film:
            gamma_f = self.film_gamma(q).unsqueeze(-1).unsqueeze(-1)
            beta_f = self.film_beta(q).unsqueeze(-1).unsqueeze(-1)
            c = c * (1.0 + gamma_f) + beta_f
        c = self.c_norm(c)

        c_flat = c.permute(0, 2, 3, 1).reshape(B, -1, self.ch)  # (B, P, ch)
        if scramble_state:
            perm = torch.randperm(c_flat.shape[1], device=device)
            c_flat = c_flat[:, perm]
        b_content = self.content_head(c_flat)                   # (B, P, cd)

        h = self.h_init(q).reshape(B, M, d)
        if self.deterministic_phase:
            theta = torch.zeros(B, M, device=device)
        else:
            theta = 2 * torch.pi * torch.rand(B, M, device=device)

        beta_val = self.log_beta.exp()

        traces = (
            {'phase': [], 'gates': [], 'attn': [], 'h': []}
            if return_trace else None
        )

        # uniform attention so T=0 still reads *something* (T-ablations)
        P = c_flat.shape[1]
        attn = torch.full((B, M, P), 1.0 / P, device=device)

        for _ in range(T):

            # 1. phase-derived gates
            g = self._gates(theta)                              # (B, M, M)

            # 2. phase-gated messages (mean over senders, self included)
            msg_all = self.msg_proj(h)                          # (B, M, msg)
            msg = torch.einsum('bij,bjk->bik', g, msg_all) / M  # (B, M, msg)

            # 3. input attention
            queries = F.normalize(self.attn_query(h), dim=-1)   # (B, M, ch)
            keys = F.normalize(c_flat, dim=-1)                  # (B, P, ch)
            attn = F.softmax(
                beta_val * torch.einsum('bmc,bpc->bmp', queries, keys),
                dim=-1,
            )
            a = torch.einsum('bmp,bpc->bmc', attn, c_flat)      # (B, M, ch)

            # 4. content update
            inp = torch.cat([a, msg], dim=-1).reshape(B * M, -1)
            h = self.cell(inp, h.reshape(B * M, d)).reshape(B, M, d)

            # 5. phase update (reads the *new* h, as in the notes)
            theta = self._phase_step(theta, h)

            if return_trace:
                traces['phase'].append(theta.detach())          # type: ignore
                traces['gates'].append(g.detach())              # type: ignore
                traces['attn'].append(attn.detach())            # type: ignore
                traces['h'].append(h.detach())                  # type: ignore

        m_content = torch.einsum('bmp,bpd->bmd', attn, b_content)
        logits = self.head(torch.cat([m_content.flatten(1), q], dim=-1))

        # phase order parameter over modules + gate openness
        with torch.no_grad():
            R = torch.stack(
                [torch.cos(theta), torch.sin(theta)], dim=-1
            ).mean(dim=1).norm(dim=-1).mean().item()
            g_final = self._gates(theta)
            off = ~torch.eye(M, dtype=torch.bool, device=device)
            metrics = {
                'phase_R': R,
                'gate_offdiag': g_final[:, off].mean().item(),
            }

        return {
            'logits': logits,
            'traces': traces,
            'metrics': metrics,                                 # type: ignore
        }

    # ------------------------------------------------------------------

    @classmethod
    def from_config(
            cls,
            cfg: SortOfClevrRecurrentSyncNetConfig,
            data_cfg: SortOfClevrDataConfig,
            ) -> SortOfClevrRecurrentSyncNet:

        if cfg.encoder_cfg.name == 'patchify':
            encoder = PatchifyEncoder.from_config(
                cfg.encoder_cfg, img_size=data_cfg.img_size  # type: ignore
            )
        elif cfg.encoder_cfg.name == 'cnn':
            encoder = CNNEncoder.from_config(
                cfg.encoder_cfg, img_size=data_cfg.img_size  # type: ignore
            )
        else:
            raise ValueError(
                f'recurrent syncnet does not support encoder: '
                f'{cfg.encoder_cfg.name}'
            )

        return cls(
            encoder=encoder,
            answer_dim=C.ANSWER_SIZE,
            q_dim=C.QUESTION_SIZE,
            n_modules=cfg.n_modules,
            module_dim=cfg.module_dim,
            msg_dim=cfg.msg_dim,
            use_film=cfg.use_film,
            beta_init=cfg.beta_init,
            learn_beta=cfg.learn_beta,
            content_dim=cfg.content_dim,
            query_hidden=cfg.query_hidden,
            T=cfg.T,
            dt=cfg.dt,
            omega_init=cfg.omega_init,
            k_hidden=cfg.k_hidden,
            deterministic_phase=cfg.deterministic_phase,
            hidden_dim=cfg.hidden_dim,
        )
