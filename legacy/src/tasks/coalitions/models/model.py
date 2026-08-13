"""Assembled coalitions model + config factory.

A single model class parameterised by `gate` (the mechanism under test or a
baseline). This keeps the fairness contract obvious: swapping `model.gate`
in a sweep changes only the gate module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ....core.config import ModelConfig
from ..config import CoalitionsDataConfig
from .base import CoalitionsBase


@dataclass
class CoalitionsConfig(ModelConfig):
    name: str = 'coalitions'

    # gate mechanism: the sole architectural variable
    #   'phase' | 'mlp' | 'recurrent' | 'attention'
    #   'no_comm' | 'always_on' | 'oracle'   (diagnostics)
    gate: str = 'phase'

    # shared backbone
    tok_dim: int = 32
    cmd_dim: int = 16
    hidden_dim: int = 64
    msg_dim: int = 32
    head_hidden: int = 0
    message_proj: str = 'shared'       # 'shared' (gate is sole bottleneck) | 'per_pair'

    # phase gate
    osc_dim: int = 2                   # dimension ladder knob (2 = scalar phase)
    phase_update: str = 'mlp'          # 'mlp' | 'kuramoto'
    alpha_init: float = 4.0
    bias_init: float = -1.0
    learn_sharpen: bool = True
    dt: float = 0.25
    phi_hidden: int = 64
    kappa_max: float = 2.0

    # baseline gates
    gate_hidden: int = 64              # mlp gate
    ctrl_dim: int = 64                 # recurrent gate controller
    key_dim: int = 32                  # attention gate


class CoalitionsModel(CoalitionsBase):

    @classmethod
    def from_config(
            cls,
            cfg: CoalitionsConfig,
            data_cfg: CoalitionsDataConfig,
            ) -> 'CoalitionsModel':

        gate_kwargs = {
            # phase gate
            'osc_dim': cfg.osc_dim,
            'phase_update': cfg.phase_update,
            'alpha_init': cfg.alpha_init,
            'bias_init': cfg.bias_init,
            'learn_sharpen': cfg.learn_sharpen,
            'dt': cfg.dt,
            'phi_hidden': cfg.phi_hidden,
            'kappa_max': cfg.kappa_max,
            # baselines
            'gate_hidden': cfg.gate_hidden,
            'ctrl_dim': cfg.ctrl_dim,
            'key_dim': cfg.key_dim,
        }

        from ..data import constants as C
        return cls(
            gate_kind=cfg.gate,
            n_modules=data_cfg.n_modules,
            K=data_cfg.K,
            readout_vocab=C.readout_vocab_size(data_cfg.n_modules, data_cfg.K),
            tok_dim=cfg.tok_dim,
            cmd_dim=cfg.cmd_dim,
            hidden_dim=cfg.hidden_dim,
            msg_dim=cfg.msg_dim,
            head_hidden=cfg.head_hidden,
            message_proj=cfg.message_proj,
            gate_kwargs=gate_kwargs,
        )
