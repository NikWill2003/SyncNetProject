from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from src.core.encoders import PatchifyEncoder, CNNEncoder
from src.tasks.sort_of_clevr.config import SortOfClevrDataConfig
from src.tasks.sort_of_clevr.contracts import SortOfClevrOutput
from src.tasks.sort_of_clevr.data import constants as C
from legacy_models.sort_of_clevr.syncnet_v1 import SortOfClevrSyncNetV1Config


@dataclass
class SortOfClevrSyncNetV3Config(SortOfClevrSyncNetV1Config):
    """Parameter-matched no-oscillator control.

    Inherits V1 fields for sweep compatibility. Ignored by V3:
    use_omega, init_omg, global_omg (no oscillators), competitive.
    rotor_dim only sets GroupNorm groups.
    """
    name: str = 'sort_of_clevr_syncnet_v3'

Encoder = PatchifyEncoder | CNNEncoder

"""SyncNet V3: parameter-matched no-oscillator control.

Hypothesis under test
---------------------
The recurring negative result (T=0 random phases matching trained
dynamics; freeze_L0 winning) suggests V1's performance comes from its
*architectural scaffolding* -- recurrent conv bottom, question-initialised
modules, attention routing, top-down feedback -- rather than from
oscillator dynamics. V3 keeps the scaffolding and removes the oscillators:

  - bottom state is an unconstrained feature map updated by a bounded
    residual recurrence (same conv, same GroupNorm groups, tanh-bounded
    update) instead of Kuramoto steps on the rotor sphere;
  - modules are plain vectors with cosine-similarity softmax routing
    instead of rotor-coherence routing;
  - no unit-norm constraint, no tangent projection, no omega rotation
    (omega is the only parameter difference vs V1: ~R*2 params).

If V3 matches V1 (IID and OOD), the scaffolding hypothesis is confirmed
head-to-head at matched parameters and matched T-step compute. If V1
beats V3 anywhere, that gap is the cleanest measure of what the
dynamics contribute.
"""


class SortOfClevrSyncNetV3(nn.Module):

    has_rotors = False  # metric/vis callbacks skip phase-specific readouts
    is_syncnet = True

    def __init__(
            self,
            encoder: Encoder,
            answer_dim: int,
            q_dim: int,
            rotor_dim: int = 4,          # kept only to define GroupNorm groups
            use_film: bool = True,
            ksize: int = 3,
            n_modules: int = 4,
            content_dim: int = 8,
            query_hidden: int = 64,
            hidden_dim: int = 128,
            T: int = 6,
            gamma: float = 0.5,
            dt: float = 0.1,
            beta_init: float = 5.0,
            learn_beta: bool = True,
            use_top_down: bool = True,
            top_down_alpha_init: float = 0.1,
            ):
        super().__init__()

        ch = encoder.ch
        assert ch % rotor_dim == 0, 'ch must be divisible by rotor_dim'

        self.ch = ch
        self.R = ch // rotor_dim
        self.answer_dim = answer_dim
        self.q_dim = q_dim
        self.use_film = use_film
        self.n_modules = n_modules
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim
        self.T = T
        self.gamma = gamma
        self.dt = dt
        self.use_top_down = use_top_down

        self.encoder = encoder
        self.spatial = encoder.spatial
        self.n_tokens = encoder.n_tokens

        if use_film:
            self.film_gamma = nn.Linear(q_dim, ch)
            self.film_beta = nn.Linear(q_dim, ch)

        # matched to AKOrNBottom's layer inventory
        self.connectivity = nn.Conv2d(ch, ch, ksize, padding=ksize // 2)
        self.c_norm = nn.GroupNorm(self.R, ch, affine=True)
        self.update_norm = nn.GroupNorm(self.R, ch, affine=True)

        self.query_head = nn.Sequential(
            nn.Linear(q_dim, query_hidden),
            nn.GELU(),
            nn.Linear(query_hidden, n_modules * ch),
        )
        self.content_head = nn.Linear(ch, content_dim)

        log_beta = torch.tensor(float(np.log(beta_init)))
        if learn_beta:
            self.log_beta = nn.Parameter(log_beta)
        else:
            self.register_buffer('log_beta', log_beta)

        if use_top_down:
            self.top_down_proj = nn.Linear(ch, ch)
            self.top_down_alpha = nn.Parameter(
                torch.tensor(top_down_alpha_init)
            )

        self.head = nn.Sequential(
            nn.Linear(n_modules * content_dim + q_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, answer_dim),
        )

    def forward(
            self,
            images: torch.Tensor,
            questions: torch.Tensor,
            t_override: Optional[int] = None,
            return_trace: bool = False,
            scramble_state: bool = False,
            **kwargs
            ) -> SortOfClevrOutput:

        x_img = images
        q = questions.float()

        B = x_img.shape[0]
        H = W = self.spatial
        ch = self.ch
        M = self.n_modules
        T = t_override if t_override is not None else self.T
        device = x_img.device

        c_input = self.encoder(x_img)

        if self.use_film:
            gamma_f = self.film_gamma(q).unsqueeze(-1).unsqueeze(-1)
            beta_f = self.film_beta(q).unsqueeze(-1).unsqueeze(-1)
            c_modulated = c_input * (1.0 + gamma_f) + beta_f
        else:
            c_modulated = c_input

        c = self.c_norm(c_modulated)

        c_flat = c.permute(0, 2, 3, 1).reshape(B, H * W, ch)
        b_content = self.content_head(c_flat)

        # state: random init, matched to V1's random rotor init
        state = torch.randn(B, ch, H, W, device=device)

        m_vec = self.query_head(q).reshape(B, M, ch)
        m_content = torch.zeros(B, M, self.content_dim, device=device)

        traces = (
            {'state': [], 'm_vec': [], 'm_content': [], 'attn': []}
            if return_trace else None
        )

        last_attn = None
        beta_val = self.log_beta.exp()

        perm = (
            torch.randperm(H * W, device=device)
            if scramble_state else None
        )

        for _ in range(T):

            if self.use_top_down and last_attn is not None:
                proj = self.top_down_proj(m_vec)
                td = torch.einsum('bmp,bmc->bpc', last_attn, proj)
                td = td.transpose(1, 2).reshape(B, ch, H, W)
                c_step = c + self.top_down_alpha * td
            else:
                c_step = c

            # bounded residual recurrence (no sphere, no rotation)
            y = self.connectivity(state) + c_step
            state = state + self.gamma * torch.tanh(self.update_norm(y))

            state_flat = state.permute(0, 2, 3, 1).reshape(B, H * W, ch)
            if perm is not None:
                state_flat = state_flat[:, perm]

            # cosine-similarity routing, matched to rotor coherence
            coh = torch.einsum(
                'bmc,bpc->bmp',
                F.normalize(m_vec, dim=-1),
                F.normalize(state_flat, dim=-1),
            )
            attn = F.softmax(beta_val * coh, dim=-1)

            drive = torch.einsum('bmp,bpc->bmc', attn, state_flat)
            m_vec = m_vec + self.dt * (drive - m_vec)

            m_content = torch.einsum('bmp,bpd->bmd', attn, b_content)

            last_attn = attn

            if return_trace:
                traces['state'].append(state.detach())      # type: ignore
                traces['m_vec'].append(m_vec.detach())      # type: ignore
                traces['m_content'].append(m_content.detach())  # type: ignore
                traces['attn'].append(attn.detach())        # type: ignore

        flat = m_content.flatten(1)
        logits = self.head(torch.cat([flat, q], dim=-1))

        return {'logits': logits, 'traces': traces}

    @classmethod
    def from_config(
            cls,
            cfg: SortOfClevrSyncNetV3Config,
            data_cfg: SortOfClevrDataConfig,
            ) -> SortOfClevrSyncNetV3:

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
                f'sort-of-clevr syncnet-v3 does not '
                f'support the encoder: {cfg.encoder_cfg.name}'
            )

        return cls(
            encoder=encoder,
            answer_dim=C.ANSWER_SIZE,
            q_dim=C.QUESTION_SIZE,
            rotor_dim=cfg.rotor_dim,
            use_film=cfg.use_film,
            ksize=cfg.ksize,
            n_modules=cfg.n_modules,
            content_dim=cfg.content_dim,
            query_hidden=cfg.query_hidden,
            hidden_dim=cfg.hidden_dim,
            T=cfg.T,
            gamma=cfg.gamma,
            dt=cfg.dt,
            beta_init=cfg.beta_init,
            learn_beta=cfg.learn_beta,
            use_top_down=cfg.use_top_down,
            top_down_alpha_init=cfg.top_down_alpha_init,
        )
