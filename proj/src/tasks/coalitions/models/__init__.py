from ....core.registry import ModelSpec

from .base import CoalitionsBase
from .model import CoalitionsModel, CoalitionsConfig
from .gates import GATES, build_gate

MODELS: dict[str, ModelSpec] = {
    'coalitions': ModelSpec(
        config=CoalitionsConfig,
        model_class=CoalitionsModel,
    ),
}

__all__ = [
    'MODELS',
    'CoalitionsBase',
    'CoalitionsModel',
    'CoalitionsConfig',
    'GATES',
    'build_gate',
]
