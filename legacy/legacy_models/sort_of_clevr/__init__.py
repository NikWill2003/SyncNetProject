"""Superseded Sort-of-CLEVR models, kept for reproducibility.

These produced results reported before the dataset corrections and the
move to the unified `syncnet.py`:

  syncnet_v1        AKOrN rotor field; Kuramoto over spatial locations
  syncnet_v2        v1 + CTC gain routing, gated content integration
  syncnet_v3        parameter-matched no-oscillator control (best of the
                    three: 0.778 vs v1 0.693 on the pre-correction data)
  recurrent_syncnet phase-as-module-state; also provided ctm_syncnet via
                    readout_mode='sync'

They are NOT registered in MODELS. To rerun one, import it here and add
a ModelSpec back to models/__init__.py. Note their numbers came from the
uncorrected generator (constant-answer non-relational subtypes, the 999
nearest-neighbour sentinel), so they are not comparable to anything
produced after that fix without a rerun.
"""

from .recurrent_syncnet import (
    SortOfClevrRecurrentSyncNet,
    SortOfClevrRecurrentSyncNetConfig,
)
from legacy_models.sort_of_clevr.syncnet_v1 import SortOfClevrSyncNetV1, SortOfClevrSyncNetV1Config
from .syncnet_v2 import SortOfClevrSyncNetV2, SortOfClevrSyncNetV2Config
from .syncnet_v3 import SortOfClevrSyncNetV3, SortOfClevrSyncNetV3Config

__all__ = [
    'SortOfClevrRecurrentSyncNet', 'SortOfClevrRecurrentSyncNetConfig',
    'SortOfClevrSyncNetV1', 'SortOfClevrSyncNetV1Config',
    'SortOfClevrSyncNetV2', 'SortOfClevrSyncNetV2Config',
    'SortOfClevrSyncNetV3', 'SortOfClevrSyncNetV3Config',
]
