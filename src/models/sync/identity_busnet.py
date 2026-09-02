"""IdentityBusNet -- the canonical model with exactly one axis moved:
identity from exchangeable to phase-native. Per-module anchor priors give
private parameters a stable referent; private GRU cells and embeddings give
the referent somewhere to matter; the protocol (message projection,
stimulus, coupling, the bus itself) stays shared -- private minds, one
radio standard.

New measurable: anchor_shuffle permutes the identity priors (module k hunts
with module j's learned preferences) while phase_shuffle permutes the
addresses -- routing-by-name separated from routing-by-address, the
signature no exchangeable model can produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from .busnet import TOK_DIM, BusNet, BusNetConfig
from .components import PhaseNative, anchor_shuffle


@dataclass
class IdentityBusNetConfig(BusNetConfig):
    name: str = 'identity_busnet'
    # True | False | 'residual' (shared weights + zero-init per-slot deltas)
    private_cells: Any = True
    per_module_anchors: bool = True


class IdentityBusNet(BusNet):

    PHASE_OVERRIDES: ClassVar[tuple[str, ...]] = ('freeze', 'shuffle', 'zero', 'anchor_shuffle')
    supported_callbacks: ClassVar[frozenset] = frozenset({'sync'})

    def _per_module_anchors(self, cfg: IdentityBusNetConfig) -> bool:
        return cfg.per_module_anchors

    def _build_identity(self, cfg: IdentityBusNetConfig):
        return PhaseNative(self.N, TOK_DIM + self.medium.out_dim,
                           cfg.module_dim, cfg.n_modules,
                           private_cells=cfg.private_cells,
                           per_module_anchors=cfg.per_module_anchors)

    def _prior_perm(self, phase_override, batch):
        if phase_override == 'anchor_shuffle' and self.cfg.per_module_anchors:
            return anchor_shuffle(self.M, batch['images'].device)
        return None

    def _loop_override(self, phase_override):
        return None if phase_override == 'anchor_shuffle' else phase_override
