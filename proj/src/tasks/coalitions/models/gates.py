"""Communication gates for the coalitions task.

Every gate maps the modules' states to a directed gate matrix G in [0,1]^{NxN},
where G[:, i, j] is the strength of the channel from sender i to receiver j.
Only this object differs across model conditions; the module GRUs, message
projections and readout heads (in base.py) are shared, so any performance
difference is attributable to the gate, not capacity.

Common interface
----------------
    has_state : bool                      # does the gate carry state across t?
    init_state(B, device) -> Tensor|None
    gate(h_prev, cmd_emb, state, oracle) -> (G (B,N,N), extras: dict)
    update(h_new, cmd_emb, state) -> new_state

Update order (matches the spec, avoids instantaneous circular dependency):
  1. G_t = gate(h_{t-1}, cmd_t, state_t)      # from carried state / prev h
  2. messages use h_{t-1} and G_t; module states -> h_t
  3. outputs from h_t
  4. state_{t+1} = update(h_t, cmd_t, state_t)

The phase gate is the mechanism under test; the rest are baselines.
"""

from __future__ import annotations

from typing import Any, Protocol
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def _zero_diag(G: torch.Tensor) -> torch.Tensor:
    N = G.shape[-1]
    eye = torch.eye(N, device=G.device, dtype=G.dtype)
    return G * (1.0 - eye)


# ---------------------------------------------------------------------------
# Diagnostic gates (no learned parameters in the gate itself)
# ---------------------------------------------------------------------------

class GateBase(nn.Module):

    def __int__(self):
        super().__init__()

    def __init__(self) -> None:
        super().__init__()
    
    def init_state(self, B, device):
        ...
    
    def gate(
            self, h_prev, cmd_emb, state, oracle=None
            ) -> tuple[torch.Tensor, dict]:
        ...
    
    def update(self, h_new, cmd_emb, state):
        ...

class NoCommGate(nn.Module):
    """g == 0 everywhere. Lower bound: joint tasks must fail."""
    has_state = False
    kind = 'no_comm'

    def __init__(self, n_modules: int, **_):
        super().__init__()
        self.N = n_modules

    def init_state(self, B, device):
        return None

    def gate(self, h_prev, cmd_emb, state, oracle=None):
        B = h_prev.shape[0]
        G = torch.zeros(B, self.N, self.N, device=h_prev.device)
        return G, {}

    def update(self, h_new, cmd_emb, state):
        return None


class AlwaysOnGate(nn.Module):
    """g == 1 off-diagonal. Tests whether unfiltered arrival interferes."""
    has_state = False
    kind = 'always_on'

    def __init__(self, n_modules: int, **_):
        super().__init__()
        self.N = n_modules

    def init_state(self, B, device):
        return None

    def gate(self, h_prev, cmd_emb, state, oracle=None):
        B = h_prev.shape[0]
        G = _zero_diag(torch.ones(B, self.N, self.N, device=h_prev.device))
        return G, {}

    def update(self, h_new, cmd_emb, state):
        return None


class OracleGate(nn.Module):
    """g == the ground-truth required adjacency at this step. Upper bound.

    Reads the flattened undirected `oracle` (B, P) supplied per step and
    scatters it into a symmetric (B, N, N) matrix.
    """
    has_state = False
    kind = 'oracle'

    def __init__(self, n_modules: int, **_):
        super().__init__()
        self.N = n_modules
        ii, jj = [], []
        for i in range(n_modules):
            for j in range(i + 1, n_modules):
                ii.append(i)
                jj.append(j)
        self.register_buffer('pair_i', torch.tensor(ii), persistent=False)
        self.register_buffer('pair_j', torch.tensor(jj), persistent=False)

    def init_state(self, B, device):
        return None

    def gate(self, h_prev, cmd_emb, state, oracle=None):
        B = h_prev.shape[0]
        G = torch.zeros(B, self.N, self.N, device=h_prev.device)
        if oracle is None:
            return G, {}
        G[:, self.pair_i, self.pair_j] = oracle # type:ignore
        G[:, self.pair_j, self.pair_i] = oracle # type:ignore
        return G, {}

    def update(self, h_new, cmd_emb, state):
        return None


# ---------------------------------------------------------------------------
# Learned baselines
# ---------------------------------------------------------------------------

class MLPGate(nn.Module):
    """Memoryless learned gate: g_ij = sigmoid(MLP[h_i, h_j, cmd]).

    Unconstrained graph, no persistence. Directed (i, j != j, i).
    """
    has_state = False
    kind = 'mlp'

    def __init__(self, n_modules, hidden_dim, cmd_dim, gate_hidden=64, **_):
        super().__init__()
        self.N = n_modules
        self.net = _mlp([2 * hidden_dim + cmd_dim, gate_hidden, 1])

    def init_state(self, B, device):
        return None

    def gate(self, h_prev, cmd_emb, state, oracle=None):
        B, N, H = h_prev.shape
        hi = h_prev.unsqueeze(2).expand(B, N, N, H)      # sender i
        hj = h_prev.unsqueeze(1).expand(B, N, N, H)      # receiver j
        c = cmd_emb.unsqueeze(1).unsqueeze(1).expand(B, N, N, -1)
        logit = self.net(torch.cat([hi, hj, c], dim=-1)).squeeze(-1)
        return _zero_diag(torch.sigmoid(logit)), {}

    def update(self, h_new, cmd_emb, state):
        return None


class RecurrentGate(nn.Module):
    """Recurrent learned gate: a GRU controller emits the full N*N matrix.

    Unconstrained graph *and* persistent -- the strongest baseline, because it
    matches the phase gate's persistence. Any remaining gap is the structural
    (circle-metric) constraint, not persistence or capacity.
    """
    has_state = True
    kind = 'recurrent'

    def __init__(self, n_modules, hidden_dim, cmd_dim, ctrl_dim=64, **_):
        super().__init__()
        self.N = n_modules
        self.ctrl_dim = ctrl_dim
        self.cell = nn.GRUCell(n_modules * hidden_dim + cmd_dim, ctrl_dim)
        self.readout = nn.Linear(ctrl_dim, n_modules * n_modules)

    def init_state(self, B, device):
        return torch.zeros(B, self.ctrl_dim, device=device)

    def gate(self, h_prev, cmd_emb, state, oracle=None):
        B = h_prev.shape[0]
        logits = self.readout(state).view(B, self.N, self.N)
        return _zero_diag(torch.sigmoid(logits)), {}

    def update(self, h_new, cmd_emb, state):
        B = h_new.shape[0]
        inp = torch.cat([h_new.reshape(B, -1), cmd_emb], dim=-1)
        return self.cell(inp, state)


class AttentionGate(nn.Module):
    """RIM-style content routing: g_ij = sigmoid(q_j . k_i / sqrt(d)).

    Unconstrained, content-based, memoryless.
    """
    has_state = False
    kind = 'attention'

    def __init__(self, n_modules, hidden_dim, cmd_dim, key_dim=32, **_):
        super().__init__()
        self.N = n_modules
        self.scale = 1.0 / math.sqrt(key_dim)
        self.q = nn.Linear(hidden_dim + cmd_dim, key_dim)
        self.k = nn.Linear(hidden_dim + cmd_dim, key_dim)

    def init_state(self, B, device):
        return None

    def gate(self, h_prev, cmd_emb, state, oracle=None):
        B, N, H = h_prev.shape
        c = cmd_emb.unsqueeze(1).expand(B, N, -1)
        hc = torch.cat([h_prev, c], dim=-1)
        q = self.q(hc)                                   # receiver queries
        k = self.k(hc)                                   # sender keys
        logit = torch.einsum('bjd,bid->bij', q, k) * self.scale  # [b, i->j? ]
        # einsum above gives [b, receiver j, sender i]; transpose to [i, j]
        logit = logit.transpose(1, 2)                    # [b, sender i, recv j]
        return _zero_diag(torch.sigmoid(logit)), {}

    def update(self, h_new, cmd_emb, state):
        return None


# ---------------------------------------------------------------------------
# Phase gate (the mechanism under test)
# ---------------------------------------------------------------------------

class PhaseGate(nn.Module):
    """Oscillator-synchrony gate.

    State: unit oscillators z (B, N, d) on S^{d-1}. Gate:
        g_ij = sigmoid(alpha * <z_i, z_j> + b)        (sharpened cosine)
    with learnable (alpha > 0, b), optionally fixed.

    Update variants (`phase_update`):
        'mlp'      : Delta = f_phi([h, z, cmd]) predicted directly, then a
                     tangent step + renormalise (exact angle add for d = 2).
        'kuramoto' : learned omega (natural frequency) and kappa (coupling);
                     phase advances by omega plus a Kuramoto mean-field pull
                     toward the other oscillators.

    Dimension d is the ladder knob: d = 2 is scalar phase (strict circle
    clustering); larger d relaxes the frustration constraint toward
    content-attention.
    """
    has_state = True
    kind = 'phase'

    def __init__(
            self, n_modules, hidden_dim, cmd_dim,
            osc_dim=2, phase_update='mlp',
            alpha_init=4.0, bias_init=-1.0, learn_sharpen=True,
            dt=0.25, phi_hidden=64, kappa_max=2.0, **_):
        super().__init__()
        self.N = n_modules
        self.d = osc_dim
        self.mode = phase_update
        self.dt = dt
        self.kappa_max = kappa_max

        log_alpha = torch.tensor(float(math.log(alpha_init)))
        bias = torch.tensor(float(bias_init))
        if learn_sharpen:
            self.log_alpha = nn.Parameter(log_alpha)
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer('log_alpha', log_alpha)
            self.register_buffer('bias', bias)

        cond = hidden_dim + osc_dim + cmd_dim
        if phase_update == 'mlp':
            # predict a tangent update of dimension d (d=2 -> scalar angle)
            out = 1 if osc_dim == 2 else osc_dim
            self.f_phi = _mlp([cond, phi_hidden, out])
        elif phase_update == 'kuramoto':
            self.f_omega = _mlp([cond, phi_hidden, 1])
            self.f_kappa = _mlp([cond, phi_hidden, 1])
            if osc_dim > 2:
                # shared skew generator for the natural-frequency rotation
                W = torch.randn(osc_dim, osc_dim) * 0.1
                self.omega_gen = nn.Parameter(W - W.t())
        else:
            raise ValueError(f'unknown phase_update {phase_update!r}')

    # -- init --
    def init_state(self, B, device):
        z = torch.randn(B, self.N, self.d, device=device)
        return F.normalize(z, dim=-1)

    # -- gate --
    def gate(self, h_prev, cmd_emb, state, oracle=None):
        z = state                                        # (B, N, d)
        dots = torch.einsum('bid,bjd->bij', z, z)        # cos similarity
        alpha = self.log_alpha.exp()
        G = torch.sigmoid(alpha * dots + self.bias)
        return _zero_diag(G), {'z': z.detach(), 'dots': dots.detach()}

    # -- update --
    def _cond(self, h_new, z, cmd_emb):
        B, N, _ = h_new.shape
        c = cmd_emb.unsqueeze(1).expand(B, N, -1)
        return torch.cat([h_new, z, c], dim=-1)

    def update(self, h_new, cmd_emb, state):
        z = state
        if self.mode == 'mlp':
            delta = self.f_phi(self._cond(h_new, z, cmd_emb))
            if self.d == 2:
                # exact rotation by predicted angle
                ang = delta.squeeze(-1) * self.dt          # (B, N)
                cos, sin = torch.cos(ang), torch.sin(ang)
                x, y = z[..., 0], z[..., 1]
                z = torch.stack([cos * x - sin * y,
                                 sin * x + cos * y], dim=-1)
            else:
                tangent = delta - (delta * z).sum(-1, keepdim=True) * z
                z = F.normalize(z + self.dt * tangent, dim=-1)
            return z

        # kuramoto
        cond = self._cond(h_new, z, cmd_emb)
        omega = self.f_omega(cond).squeeze(-1)                     # (B, N)
        kappa = self.kappa_max * torch.sigmoid(self.f_kappa(cond).squeeze(-1))
        if self.d == 2:
            phi = torch.atan2(z[..., 1], z[..., 0])               # (B, N)
            diff = phi.unsqueeze(1) - phi.unsqueeze(2)            # phi_j - phi_i
            coupling = torch.sin(diff).mean(dim=1)                # mean over j
            phi = phi + self.dt * (omega + kappa * coupling)
            z = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
        else:
            mean_field = z.mean(dim=1, keepdim=True)              # (B,1,d)
            pull = mean_field - (mean_field * z).sum(-1, keepdim=True) * z
            rot = torch.einsum('de,bne->bnd', self.omega_gen, z)
            dz = omega.unsqueeze(-1) * rot + kappa.unsqueeze(-1) * pull
            z = F.normalize(z + self.dt * dz, dim=-1)
        return z


GATES = {
    'no_comm': NoCommGate,
    'always_on': AlwaysOnGate,
    'oracle': OracleGate,
    'mlp': MLPGate,
    'recurrent': RecurrentGate,
    'attention': AttentionGate,
    'phase': PhaseGate,
}

def build_gate(kind: str, **kwargs) -> GateBase:
    if kind not in GATES:
        raise ValueError(f'unknown gate kind {kind!r}; choices: {list(GATES)}')
    return GATES[kind](**kwargs)
