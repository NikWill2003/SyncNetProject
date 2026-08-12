from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING

from src.core.config import ModelConfig
from src.core.encoders import (
    PatchifyEncoder, CNNEncoder, EncoderConfig
    )
from src.tasks.sort_of_clevr.config import SortOfClevrDataConfig
from src.tasks.sort_of_clevr.contracts import SortOfClevrOutput
from src.tasks.sort_of_clevr.data import constants as C


@dataclass
class SortOfClevrSyncNetV1Config(ModelConfig):
    name: str = 'sort_of_clevr_syncnet_v1'

    # Rotor / bottom state
    rotor_dim: int = 4
    use_film: bool = True

    # AKOrN bottom
    ksize: int = 3
    use_omega: bool = True
    init_omg: float = 1.0
    global_omg: bool = False

    # Modules
    n_modules: int = 4
    content_dim: int = 8
    query_hidden: int = 64

    # Readout
    hidden_dim: int = 128

    # Dynamics
    T: int = 6
    gamma: float = 0.5
    dt: float = 0.1

    # Routing
    beta_init: float = 5.0
    learn_beta: bool = True
    competitive: bool = False

    # Top-down
    use_top_down: bool = True
    top_down_alpha_init: float = 0.1

    encoder_cfg: EncoderConfig = MISSING

Encoder = PatchifyEncoder | CNNEncoder


def normalize_rotors(x: torch.Tensor, n: int) -> torch.Tensor:
    B, ch, H, W = x.shape

    x_r = x.reshape(B, ch // n, n, H, W)
    norm = x_r.norm(dim=2, keepdim=True).clamp(min=1e-8)

    return (x_r / norm).reshape(B, ch, H, W)


class OmegaLayer(nn.Module):

    def __init__(
            self,
            ch: int,
            init_omg: float = 1.0,
            global_omg: bool = False,
            ):
        super().__init__()

        assert ch % 2 == 0, 'ch must be divisible by 2'

        self.ch = ch
        self.n_pairs = ch // 2
        self.global_omg = global_omg

        if global_omg:
            self.omg_param = nn.Parameter(
                init_omg * (1 / np.sqrt(2)) * torch.ones(2)
            )
        else:
            self.omg_param = nn.Parameter(
                init_omg * (1 / np.sqrt(2)) * torch.ones(self.n_pairs, 2)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, ch, H, W = x.shape

        x_pairs = x.reshape(B, self.n_pairs, 2, H, W)

        if self.global_omg:
            omg = torch.linalg.norm(self.omg_param).expand(self.n_pairs)
        else:
            omg = torch.linalg.norm(self.omg_param, dim=1)

        omg_b = omg[None, :, None, None]

        omg_x = torch.stack(
            [
                omg_b * x_pairs[:, :, 1],
                -omg_b * x_pairs[:, :, 0],
            ],
            dim=2,
        )

        return omg_x.reshape(B, ch, H, W)


class AKOrNBottom(nn.Module):

    def __init__(
            self,
            ch: int,
            rotor_dim: int,
            ksize: int = 3,
            use_omega: bool = True,
            init_omg: float = 1.0,
            global_omg: bool = False,
            ):
        super().__init__()

        assert ch % rotor_dim == 0, 'ch must be divisible by n'

        self.ch = ch
        self.n = rotor_dim
        self.R = ch // rotor_dim

        self.connectivity = nn.Conv2d(
            ch,
            ch,
            ksize,
            padding=ksize // 2,
        )

        self.c_norm = nn.GroupNorm(
            self.R,
            ch,
            affine=True,
        )

        if use_omega:
            self.omega = OmegaLayer(
                ch=ch,
                init_omg=init_omg,
                global_omg=global_omg,
            )
        else:
            self.omega = None

    def normalize_c(self, c: torch.Tensor) -> torch.Tensor:
        return self.c_norm(c)

    def step(
            self,
            x: torch.Tensor,
            c: torch.Tensor,
            gamma: float,
            ) -> torch.Tensor:
        """One Kuramoto update step on (B, ch, H, W) tensors."""
        B, ch, H, W = x.shape

        n = self.n
        R = self.R

        y = self.connectivity(x) + c

        # Per-rotor tangent projection
        y_r = y.reshape(B, R, n, H, W)
        x_r = x.reshape(B, R, n, H, W)

        dot = (y_r * x_r).sum(dim=2, keepdim=True)
        y_perp = (y_r - dot * x_r).reshape(B, ch, H, W)

        if self.omega is not None:
            drive = y_perp + self.omega(x)
        else:
            drive = y_perp

        return normalize_rotors(x + gamma * drive, n)
    
    @classmethod
    def from_config(cls, cfg: SortOfClevrSyncNetV1Config) -> AKOrNBottom:
        return cls(
            ch=cfg.encoder_cfg.ch, # type: ignore
            rotor_dim=cfg.rotor_dim,
            ksize=cfg.ksize,
            use_omega=cfg.use_omega,
            init_omg=cfg.init_omg,
            global_omg=cfg.global_omg,
        )



class SortOfClevrSyncNetV1(nn.Module):

    has_rotors = True   # metric/vis callbacks read phase-specific traces
    is_syncnet = True

    def __init__(
            self,
            encoder: Encoder,
            answer_dim: int,
            q_dim: int,

            # Channels
            rotor_dim: int = 4,

            # FiLM
            use_film: bool = True,

            # AKOrN bottom
            ksize: int = 3,
            use_omega: bool = True,
            init_omg: float = 1.0,
            global_omg: bool = False,

            # Modules
            n_modules: int = 4,
            content_dim: int = 8,
            query_hidden: int = 64,

            # Readout
            hidden_dim: int = 128,

            # Dynamics
            T: int = 6,
            gamma: float = 0.5,
            dt: float = 0.1,

            # Routing
            beta_init: float = 5.0,
            learn_beta: bool = True,
            competitive: bool = False,

            # Top-down
            use_top_down: bool = True,
            top_down_alpha_init: float = 0.1,
            ):
        super().__init__()
        
        ch = encoder.ch

        assert ch % rotor_dim == 0, 'ch must be divisible by n'

        self.ch = ch
        self.rotor_dim = rotor_dim
        self.R = ch // rotor_dim

        self.answer_dim = answer_dim
        self.q_dim = q_dim

        self.use_film = use_film

        self.n_modules = n_modules
        self.content_dim = content_dim
        self.query_hidden = query_hidden
        self.hidden_dim = hidden_dim

        self.T = T
        self.gamma = gamma
        self.dt = dt

        self.beta_init = beta_init
        self.learn_beta = learn_beta
        self.competitive = competitive

        self.use_top_down = use_top_down
        self.top_down_alpha_init = top_down_alpha_init

        # Encoder
        self.encoder = encoder
        self.spatial = self.encoder.spatial
        self.n_tokens = self.encoder.n_tokens

        # FiLM
        if self.use_film:
            self.film_gamma = nn.Linear(q_dim, ch)
            self.film_beta = nn.Linear(q_dim, ch)

        # AKOrN bottom
        self.bottom = AKOrNBottom(
            ch=ch,
            rotor_dim=rotor_dim,
            ksize=ksize,
            use_omega=use_omega,
            init_omg=init_omg,
            global_omg=global_omg,
        )

        # Modules: question -> module phase queries
        self.query_head = nn.Sequential(
            nn.Linear(q_dim, query_hidden),
            nn.GELU(),
            nn.Linear(query_hidden, n_modules * ch),
        )

        # Bottom -> content readout
        self.content_head = nn.Linear(ch, content_dim)

        # Learnable inverse temperature
        log_beta = torch.tensor(float(np.log(beta_init)))

        if learn_beta:
            self.log_beta = nn.Parameter(log_beta)
        else:
            self.register_buffer('log_beta', log_beta)

        # Top-down projection
        if self.use_top_down:
            self.top_down_proj = nn.Linear(ch, ch)
            self.top_down_alpha = nn.Parameter(
                torch.tensor(top_down_alpha_init)
            )

        # Readout head
        self.head = nn.Sequential(
            nn.Linear(n_modules * content_dim + q_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, answer_dim),
        )

    def _beta_at(self, t: int, T: int) -> torch.Tensor:
        """Routing inverse-temperature at dynamics step t. Constant in v1;
        variants override to implement schedules."""
        return self.log_beta.exp()

    def _route(
            self,
            coh: torch.Tensor,
            beta: torch.Tensor,
            ) -> torch.Tensor:
        """Coherence (B, M, P) -> routing weights (B, M, P). Soft(max) in v1;
        variants override for discrete/competitive routing."""
        if self.competitive:
            attn_raw = F.softmax(beta * coh, dim=1)
            return attn_raw / attn_raw.sum(
                dim=-1,
                keepdim=True,
            ).clamp(min=1e-8)

        return F.softmax(beta * coh, dim=-1)

    @staticmethod
    def _normalize_rotors_mr(x: torch.Tensor) -> torch.Tensor:
        """Per-rotor normalize on (B, M, R, n)."""
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def _module_step(
            self,
            m_phase: torch.Tensor,
            m_content: torch.Tensor,
            b_phase_r: torch.Tensor,
            b_content: torch.Tensor,
            beta: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One update of the module layer.

        m_phase:
            (B, M, R, n)

        m_content:
            (B, M, content_dim)

        b_phase_r:
            (B, P, R, n)

        b_content:
            (B, P, content_dim)
        """
        # Per-rotor coherence: sum over rotor dim of dot product over n
        coh_per_rotor = torch.einsum(
            'bmrn,bprn->bmpr',
            m_phase,
            b_phase_r,
        )

        coh = coh_per_rotor.sum(-1)

        # Routing
        attn = self._route(coh, beta)

        # Phase update
        drive = torch.einsum(
            'bmp,bprn->bmrn',
            attn,
            b_phase_r,
        )

        dot = (drive * m_phase).sum(-1, keepdim=True)
        drive_perp = drive - dot * m_phase

        m_phase_new = self._normalize_rotors_mr(
            m_phase + self.dt * drive_perp
        )

        # Content update
        m_content_new = torch.einsum(
            'bmp,bpd->bmd',
            attn,
            b_content,
        )

        return m_phase_new, m_content_new, attn

    def forward(
            self,
            images: torch.Tensor,
            questions: torch.Tensor,
            t_override: Optional[int] = None,
            return_trace: bool = False,
            scramble_phase: bool = False,
            **kwargs
            ) -> SortOfClevrOutput:
        
        x = images
        q = questions.float()

        B = x.shape[0]
        H = W = self.spatial
        device = x.device

        ch = self.ch
        n = self.rotor_dim
        R = self.R
        M = self.n_modules
        T = t_override if t_override is not None else self.T

        # Encode
        c_input = self.encoder(x)

        # FiLM
        if self.use_film:
            gamma_f = self.film_gamma(q).unsqueeze(-1).unsqueeze(-1)
            beta_f = self.film_beta(q).unsqueeze(-1).unsqueeze(-1)

            c_modulated = c_input * (1.0 + gamma_f) + beta_f
        else:
            c_modulated = c_input

        # Conditional stimulus
        c = self.bottom.normalize_c(c_modulated)

        # Bottom content
        c_flat = c.permute(0, 2, 3, 1).reshape(B, H * W, ch)
        b_content = self.content_head(c_flat)

        # Init bottom rotors
        rotors = torch.randn(B, ch, H, W, device=device)
        rotors = normalize_rotors(rotors, n)

        # Init modules from question
        m_phase = self.query_head(q).reshape(B, M, R, n)
        m_phase = self._normalize_rotors_mr(m_phase)

        m_content = torch.zeros(
            B,
            M,
            self.content_dim,
            device=device,
        )

        traces = (
            {
                'rotors': [],
                'm_phase': [],
                'm_content': [],
                'attn': [],
            }
            if return_trace else None
        )

        last_attn = None

        perm = (
            torch.randperm(H * W, device=device)
            if scramble_phase else None
        )

        for t_step in range(T):

            beta_val = self._beta_at(t_step, T)

            # Top-down feedback
            if self.use_top_down and last_attn is not None:
                m_phase_flat = m_phase.reshape(B, M, ch)

                proj = self.top_down_proj(m_phase_flat)

                td_signal = torch.einsum(
                    'bmp,bmc->bpc',
                    last_attn,
                    proj,
                )

                td_signal = td_signal.transpose(1, 2).reshape(B, ch, H, W)

                c_step = c + self.top_down_alpha * td_signal
            else:
                c_step = c

            # Bottom step
            rotors = self.bottom.step(
                rotors,
                c_step,
                self.gamma,
            )

            # Module step
            rotors_flat = rotors.permute(0, 2, 3, 1).reshape(B, H * W, ch)
            if perm is not None:
                rotors_flat = rotors_flat[:, perm]
            rotors_r = rotors_flat.reshape(B, H * W, R, n)

            m_phase, m_content, attn = self._module_step(
                m_phase,
                m_content,
                rotors_r,
                b_content,
                beta_val,
            )

            last_attn = attn

            if return_trace:
                traces['rotors'].append(rotors.detach()) # type: ignore
                traces['m_phase'].append(m_phase.detach()) # type: ignore
                traces['m_content'].append(m_content.detach()) # type: ignore
                traces['attn'].append(attn.detach()) # type: ignore

        # Readout
        flat = m_content.flatten(1)
        logits = self.head(torch.cat([flat, q], dim=-1))

        return {
            'logits': logits,
            'traces': traces
            }
    
    @classmethod
    def from_config(
            cls,
            cfg: SortOfClevrSyncNetV1Config,
            data_cfg: SortOfClevrDataConfig,
            ) -> SortOfClevrSyncNetV1:
        
        if cfg.encoder_cfg.name == 'patchify':
            encoder = PatchifyEncoder.from_config(
                cfg.encoder_cfg, img_size=data_cfg.img_size # type: ignore
                )
            
        elif cfg.encoder_cfg.name == 'cnn':
            encoder = CNNEncoder.from_config(
                cfg.encoder_cfg, img_size=data_cfg.img_size # type: ignore
            )
        else:
            raise ValueError(
                f'sort-of-clevr syncnet-v1 does not'
                f'support the encoder: {cfg.encoder_cfg.name}'
                )
        
        return cls(
            encoder=encoder,
            answer_dim=C.ANSWER_SIZE,
            q_dim=C.QUESTION_SIZE,

            rotor_dim=cfg.rotor_dim,

            use_film=cfg.use_film,

            ksize=cfg.ksize,
            use_omega=cfg.use_omega,
            init_omg=cfg.init_omg,
            global_omg=cfg.global_omg,

            n_modules=cfg.n_modules,
            content_dim=cfg.content_dim,
            query_hidden=cfg.query_hidden,

            hidden_dim=cfg.hidden_dim,

            T=cfg.T,
            gamma=cfg.gamma,
            dt=cfg.dt,

            beta_init=cfg.beta_init,
            learn_beta=cfg.learn_beta,
            competitive=cfg.competitive,

            use_top_down=cfg.use_top_down,
            top_down_alpha_init=cfg.top_down_alpha_init,
        )