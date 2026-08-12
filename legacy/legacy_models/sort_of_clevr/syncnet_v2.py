from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.sort_of_clevr.config import SortOfClevrDataConfig
from legacy_models.sort_of_clevr.syncnet_v1 import SortOfClevrSyncNetV1, SortOfClevrSyncNetV1Config


@dataclass
class SortOfClevrSyncNetV2Config(SortOfClevrSyncNetV1Config):
    """V1 config + CTC gain routing and gated content integration.

    All V1 fields apply; `competitive` is ignored (V2's gain routing
    replaces the softmax entirely).
    """
    name: str = 'sort_of_clevr_syncnet_v2'

"""SyncNet V2: communication-through-coherence routing.

Hypothesis under test
---------------------
V1 routes content with a competitive softmax over phase coherence. With
P ~ 200 patches, softmax saturates: small coherence differences either
vanish (low beta) or collapse to near one-hot (high beta), so graded
phase information has a very narrow channel through which to influence
content. If that bottleneck is why trained dynamics fail to beat T=0,
then replacing the softmax with a *gain* (multiplicative, non-competitive,
bounded per-patch weight -- Fries-style CTC) should recover a benefit
from the dynamics.

Differences from V1 (everything else identical):
  1. Routing: per-patch sigmoid gain on coherence, normalised to sum to 1
     (soft multi-selection) instead of softmax (competitive selection).
  2. Content: gated integration across T steps (GRU-style scalar gate per
     module) instead of overwriting m_content each step -- so content can
     accumulate along a synchrony trajectory rather than reflecting only
     the final routing state.
"""


class SortOfClevrSyncNetV2(SortOfClevrSyncNetV1):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # scalar gate per module from [old content, aggregated content]
        self.content_gate = nn.Sequential(
            nn.Linear(2 * self.content_dim, self.content_dim),
            nn.GELU(),
            nn.Linear(self.content_dim, 1),
        )

    def _module_step(
            self,
            m_phase: torch.Tensor,
            m_content: torch.Tensor,
            b_phase_r: torch.Tensor,
            b_content: torch.Tensor,
            beta: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Per-rotor coherence, as in V1
        coh = torch.einsum(
            'bmrn,bprn->bmpr',
            m_phase,
            b_phase_r,
        ).sum(-1)

        # CTC gain routing: bounded per-patch gain, then normalise.
        # Centre coherence per module so the gain is scale-free.
        coh_centred = coh - coh.mean(dim=-1, keepdim=True)
        gain = torch.sigmoid(beta * coh_centred)
        attn = gain / gain.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Phase update (identical to V1)
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

        # Gated content integration (V1 overwrites)
        agg = torch.einsum(
            'bmp,bpd->bmd',
            attn,
            b_content,
        )
        z = torch.sigmoid(
            self.content_gate(torch.cat([m_content, agg], dim=-1))
        )
        m_content_new = (1.0 - z) * m_content + z * agg

        return m_phase_new, m_content_new, attn

    @classmethod
    def from_config(
            cls,
            cfg: SortOfClevrSyncNetV2Config,
            data_cfg: SortOfClevrDataConfig,
            ) -> SortOfClevrSyncNetV2:
        # V1's from_config builds cls(...), so inheritance handles this,
        # but keep the override explicit for clarity of dispatch.
        return super().from_config(cfg, data_cfg)  # type: ignore
