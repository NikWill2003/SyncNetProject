"""The canonical field-bus SyncNet on SQOOP.

The inner model is src/models/busnet.py unchanged: the oscillator-field
front end is task-agnostic (64x64 -> a 16x16 map), the phase slots, the
vector-phase bus and the head readout carry over as they are. What the
wrapper supplies is the question encoding -- SQOOP questions are three
indices [x, rel, y] into the joint 40-token vocabulary, encoded here as
three concatenated one-hots (q_size = 120) so the conditioning vector is
parameter-free and deterministic -- and the answer dimension (2). There is
no object tokenizer on SQOOP (letters are not colour-keyed), so the inner
model runs with object_colours=None: the objects front end is unavailable
and the object-identity metrics are skipped; phase_R, n_clusters_eff and
the read diagnostics remain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ....models.busnet import BusNet, BusNetConfig
from ..config import SqoopDataConfig
from ..contracts import SqoopBatch, SqoopOutput
from ..data import constants as C


@dataclass
class SqoopBusNetConfig(BusNetConfig):
    name: str = 'sqoop_busnet'
    q_size: int = C.QUESTION_LEN * C.VOCAB_SIZE          # 3 x 40 one-hots
    encoder: dict[str, Any] = field(default_factory=lambda: {'name': 'field'})
    per_module_gru: bool = False
    phase_repr: str = 'vector'
    osc_dim: int = 6
    drive: str = 'stimulus'


class SqoopBusNet(nn.Module):

    is_syncnet = True
    SUPPORTED_OVERRIDES = BusNet.SUPPORTED_OVERRIDES
    GATE_OVERRIDES = BusNet.GATE_OVERRIDES
    PHASE_OVERRIDES = BusNet.PHASE_OVERRIDES

    def __init__(self, inner: BusNet):
        super().__init__()
        self.inner = inner

    def forward(self, batch: SqoopBatch, **overrides) -> SqoopOutput:
        q = F.one_hot(batch['questions'].long(), C.VOCAB_SIZE).float().flatten(1)   # (B, 120)
        return self.inner(batch['images'], q, **overrides)  # type: ignore[return-value]

    @classmethod
    def from_config(cls, cfg: SqoopBusNetConfig, data_cfg: SqoopDataConfig) -> 'SqoopBusNet':
        inner = BusNet(cfg, int(data_cfg.img_size), C.ANSWER_SIZE, object_colours=None)
        return cls(inner)
