"""M -- the communication substrate. SharedBus is the reference
implementation of the thesis; every other medium exists to be compared
against it.

SharedBus: one wire, Bus = sum_j m_j z_j^T; each row reads the sum through
its own orthonormal frame (first vector its phase, the rest Gram-Schmidt of
fixed reference directions), own echo subtracted, divided by N. Nothing but
phase can select a sender here: L2 engineered into physics. Capacity is
phase_dim orthogonal directions; interference is the price of promiscuity.

PrivateLines: per-sender access with a gate -- the gated model's medium and
the suite's upper controls (gate='full' is the bandwidth ceiling;
gate='attn' is the content-routing competitor whose victory over the phase
gate is the L2 exhibit; gate='phase' is the scalar (1+cos)/2 gate
generalised to S^{d-1}).

SilentBus: carries nothing -- the medium term of the question-only floor.
Address modes (computed / static / open) are the model's concern, not the
medium's: they choose z, the medium only consumes it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SharedBus(nn.Module):

    def __init__(self, module_dim: int, msg_dim: int, phase_dim: int, ref_seed: int = 1234):
        super().__init__()
        self.msg_dim, self.phase_dim = msg_dim, phase_dim
        self.msg_proj = nn.Linear(module_dim, msg_dim)
        g = torch.Generator().manual_seed(ref_seed)
        self.register_buffer('frame_ref',
                             F.normalize(torch.randn(phase_dim - 1, phase_dim, generator=g), dim=-1))

    @property
    def out_dim(self) -> int:
        return self.msg_dim * self.phase_dim

    def _frame(self, z: Tensor) -> Tensor:
        """(B, N, d) -> (B, N, d, d): row i's frame, first vector z_i, the
        rest Gram-Schmidt of fixed reference directions."""
        vecs = [z]
        for k in range(self.phase_dim - 1):
            v = self.frame_ref[k].to(z.dtype).expand_as(z)
            for u in vecs:
                v = v - (v * u).sum(-1, keepdim=True) * u
            vecs.append(F.normalize(v, dim=-1))
        return torch.stack(vecs, 2)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        B, N, _ = h.shape
        m = self.msg_proj(h)                                                 # (B, N, D)
        bus = torch.einsum('bnD,bnd->bDd', m, z)                             # one wire, shared
        Fr = self._frame(z)                                                  # (B, N, d, d)
        r = torch.einsum('bDd,bnad->bnDa', bus, Fr)
        r = r - torch.einsum('bnD,bnd,bnad->bnDa', m, z, Fr)                 # echo cancellation
        return r.flatten(2) / float(N)


class SilentBus(SharedBus):
    """No medium: the floor. Same output shape so the cell is unchanged."""

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        B, N, _ = h.shape
        return torch.zeros(B, N, self.out_dim, device=h.device, dtype=h.dtype)


class PrivateLines(nn.Module):
    """Per-sender access restored, with a gate deciding who is heard.

    gate='full'  every sender's message concatenated at every receiver
    gate='attn'  content attention over senders (the L2 competitor)
    gate='phase' g_ij = (1 + <z_i, z_j>) / 2, the scalar gate on S^{d-1}
    """

    def __init__(self, module_dim: int, msg_dim: int, phase_dim: int,
                 n_rows: int, gate: str = 'attn', attn_dim: int = 32):
        super().__init__()
        assert gate in ('full', 'attn', 'phase')
        self.gate, self.n_rows, self.msg_dim = gate, n_rows, msg_dim
        self.msg_proj = nn.Linear(module_dim, msg_dim)
        if gate == 'attn':
            self.q_proj = nn.Linear(module_dim, attn_dim)
            self.k_proj = nn.Linear(module_dim, attn_dim)
            self.scale = attn_dim ** -0.5

    @property
    def out_dim(self) -> int:
        return self.msg_dim * self.n_rows if self.gate == 'full' else self.msg_dim

    def forward(self, h: Tensor, z: Tensor, gate_override: str | None = None) -> Tensor:
        B, N, _ = h.shape
        m = self.msg_proj(h)                                                 # (B, N, D)
        eye = torch.eye(N, device=h.device, dtype=torch.bool)
        if gate_override == 'zero':
            B_, N_ = h.shape[:2]
            return torch.zeros(B_, N_, self.out_dim, device=h.device, dtype=h.dtype)
        if gate_override == 'open' and self.gate != 'full':
            w = (1.0 - eye.float())[None].expand(B, N, N) / max(N - 1, 1)
            return torch.einsum('bnm,bmD->bnD', w.to(m.dtype), m)
        if self.gate == 'full':
            r = m.unsqueeze(1).expand(B, N, N, self.msg_dim).clone()
            r = r.masked_fill(eye[None, :, :, None], 0.0)                    # no self line
            return r.flatten(2)
        if self.gate == 'attn':
            logits = torch.einsum('bnd,bmd->bnm', self.q_proj(h), self.k_proj(h)) * self.scale
            logits = logits.masked_fill(eye[None], float('-inf'))
            w = torch.softmax(logits, dim=-1)
        else:                                                                # phase
            w = (1 + torch.einsum('bnd,bmd->bnm', z, z)) / 2
            w = w.masked_fill(eye[None], 0.0) / max(N - 1, 1)
        return torch.einsum('bnm,bmD->bnD', w, m)
