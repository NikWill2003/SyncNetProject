"""Shared backbone for the coalitions models.

All model conditions share this backbone (module GRUs, message projections,
local readout heads) and differ only in the injected gate module. This is the
fairness constraint made structural: the gate is the sole architectural
variable.

forward() returns {'logits': (B, T, N, K), 'traces': {...}} following the repo
convention. Traces expose, per timestep, the directed gate matrix and (for the
phase gate) the oscillator states -- everything the metric callbacks need.

Intervention hooks
------------------
`gate_transform(G, t, state)` and `state_transform(state, t)` are optional
callables applied each step, used by the causal-analysis callbacks for forced
(de)synchronisation, gate clamping and phase-offset sweeps. They default to
None (no intervention) and are never serialised into the config.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from ..data import constants as C
from ..contracts import CoalitionsOutput
from .gates import build_gate


class CoalitionsBase(nn.Module):

    is_coalitions = True

    def __init__(
            self,
            gate_kind: str,
            n_modules: int = 4,
            K: int = 16,
            readout_vocab: int | None = None,
            n_commands: int = C.N_COMMANDS,
            tok_dim: int = 32,
            cmd_dim: int = 16,
            hidden_dim: int = 64,
            msg_dim: int = 32,
            head_hidden: int = 0,
            message_proj: str = 'shared',
            gate_kwargs: Optional[dict] = None,
            ):
        super().__init__()
        self.N = n_modules
        self.K = K
        self.out_vocab = (
            readout_vocab if readout_vocab is not None
            else C.readout_vocab_size(n_modules, K)
        )
        self.message_proj = message_proj
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.gate_kind = gate_kind

        self.tok_embed = nn.Embedding(C.vocab_size(K), tok_dim)
        self.cmd_embed = nn.Embedding(n_commands, cmd_dim)

        # per-module GRUs (separate, so modules can specialise)
        in_dim = tok_dim + cmd_dim + msg_dim
        self.cells = nn.ModuleList(
            [nn.GRUCell(in_dim, hidden_dim) for _ in range(n_modules)]
        )

        # message value projection. Source = [current token embedding ; prev
        # hidden state], so a message carries the sender's CURRENT observation
        # (needed for the current-step combine) with no circular dependency.
        # 'shared' makes the value sender-generic, so the gate G_ij is the ONLY
        # lever selecting which sender reaches receiver j -- a true bottleneck
        # (as in attention). 'per_pair' gives each ordered pair its own
        # subspace, which lets a receiver content-route AROUND the gate; useful
        # as an ablation but NOT valid for the synchrony experiment.
        val_in = tok_dim + hidden_dim
        if message_proj == 'shared':
            self.Wv = nn.Parameter(
                torch.randn(msg_dim, val_in) / (val_in ** 0.5)
            )
            self.bv = nn.Parameter(torch.zeros(msg_dim))
        elif message_proj == 'per_pair':
            self.Wv = nn.Parameter(
                torch.randn(n_modules, n_modules, msg_dim, val_in)
                / (val_in ** 0.5)
            )
            self.bv = nn.Parameter(torch.zeros(n_modules, n_modules, msg_dim))
        else:
            raise ValueError(f'unknown message_proj {message_proj!r}')

        # local readout heads (own hidden state only)
        if head_hidden > 0:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, head_hidden),
                    nn.GELU(),
                    nn.Linear(head_hidden, self.out_vocab),
                ) for _ in range(n_modules)
            ])
        else:
            self.heads = nn.ModuleList(
                [nn.Linear(hidden_dim, self.out_vocab)
                 for _ in range(n_modules)]
            )

        gk = dict(gate_kwargs or {})
        self.gate = build_gate(
            gate_kind,
            n_modules=n_modules,
            hidden_dim=hidden_dim,
            cmd_dim=cmd_dim,
            **gk,
        )

    def _messages(self, tok: torch.Tensor, h_prev: torch.Tensor,
                  G: torch.Tensor) -> torch.Tensor:
        u = torch.cat([tok, h_prev], dim=-1)          # (B, N, tok+hidden)
        if self.message_proj == 'shared':
            V = torch.einsum('mh,bih->bim', self.Wv, u) + self.bv
            return torch.einsum('bij,bim->bjm', G, V)
        V = torch.einsum('ijmh,bih->bijm', self.Wv, u) + self.bv
        return torch.einsum('bij,bijm->bjm', G, V)

    def forward(self, batch: dict, **overrides) -> CoalitionsOutput:
        """The Trainer's calling convention: one positional batch dict.

        Every other task in this repo goes through a task adapter for this;
        coalitions has no separate shared model to adapt, so the seam lives
        here. `forward_seq` keeps the tensor-argument signature that the
        analysis callbacks want (traces, gate_transform, state_transform),
        and this wrapper is the only thing the Trainer sees.
        """
        return self.forward_seq(
            batch['streams'],
            batch['commands'],
            oracle_adj=batch.get('oracle_adj'),
            **overrides,
        )

    def forward_seq(
            self,
            streams: torch.Tensor,          # (B, T, N) long
            commands: torch.Tensor,         # (B, T)    long
            oracle_adj: Optional[torch.Tensor] = None,   # (B, T, P) float
            return_trace: bool = False,
            gate_transform: Optional[Callable] = None,
            state_transform: Optional[Callable] = None,
            **kwargs,
            ) -> CoalitionsOutput:
        B, T, N = streams.shape
        device = streams.device
        H = self.hidden_dim

        h = torch.zeros(B, N, H, device=device)
        state = self.gate.init_state(B, device) 

        logits_seq = []
        tr_gate, tr_z = [], []

        for t in range(T):
            tok = self.tok_embed(streams[:, t, :])        # (B, N, tok)
            cmd = self.cmd_embed(commands[:, t])          # (B, cmd)

            oracle_t = None if oracle_adj is None else oracle_adj[:, t, :]

            if state_transform is not None and state is not None:
                state = state_transform(state, t)

            G, extras = self.gate.gate(h, cmd, state, oracle=oracle_t)
            if gate_transform is not None:
                G = gate_transform(G, t, state)

            m = self._messages(tok, h, G)                 # (B, N, msg)

            cmd_b = cmd.unsqueeze(1).expand(B, N, -1)
            inp = torch.cat([tok, cmd_b, m], dim=-1)      # (B, N, in_dim)

            h_new = torch.empty_like(h)
            for k in range(N):
                h_new[:, k] = self.cells[k](inp[:, k], h[:, k])
            h = h_new

            step_logits = torch.stack(
                [self.heads[k](h[:, k]) for k in range(N)], dim=1
            )                                             # (B, N, K)
            logits_seq.append(step_logits)

            if self.gate.has_state:
                state = self.gate.update(h, cmd, state)

            if return_trace:
                tr_gate.append(G.detach())
                if 'z' in extras:
                    tr_z.append(extras['z'])

        logits = torch.stack(logits_seq, dim=1)           # (B, T, N, K)

        traces = None
        if return_trace:
            traces = {'gate': torch.stack(tr_gate, dim=1)}   # (B, T, N, N)
            if tr_z:
                traces['z'] = torch.stack(tr_z, dim=1)       # (B, T, N, d)

        return {'logits': logits, 'traces': traces}
