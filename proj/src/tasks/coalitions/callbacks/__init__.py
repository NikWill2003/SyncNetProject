from ....core.registry import CallbackSpec
from .metrics import coalitions_metric_callbacks, CoalitionsMetricsCallbackCfg

coalitions_callbacks: dict[str, CallbackSpec] = {
    **coalitions_metric_callbacks,
}

__all__ = [
    'coalitions_callbacks',
    'CoalitionsMetricsCallbackCfg',
]
