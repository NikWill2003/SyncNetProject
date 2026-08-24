from .metrics import (
    sort_of_clevr_metric_callbacks,
    AccuracyCallbackCfg,
    QtypeAccuracyCallbackCfg,
    SubtypeAccuracyCallbackCfg,
)
from .visualisations import sort_of_clevr_visualisation_callbacks
from .sync_metrics import sort_of_clevr_sync_metric_callbacks
from .sync_viz import sort_of_clevr_sync_viz_callbacks
from .t_variance import sort_of_clevr_t_variance_callbacks
from .interventions import sort_of_clevr_interventions_callbacks
from .binding_analysis import sort_of_clevr_binding_analysis_callbacks

from ....core.registry import CallbackSpec

sort_of_clevr_callbacks: dict[str, CallbackSpec] = (
    sort_of_clevr_metric_callbacks
    | sort_of_clevr_visualisation_callbacks
    | sort_of_clevr_sync_metric_callbacks
    | sort_of_clevr_sync_viz_callbacks
    | sort_of_clevr_t_variance_callbacks
    | sort_of_clevr_interventions_callbacks
    | sort_of_clevr_binding_analysis_callbacks
    )

__all__ = [
    'sort_of_clevr_callbacks',
    'AccuracyCallbackCfg',
    'QtypeAccuracyCallbackCfg',
    'SubtypeAccuracyCallbackCfg',
]
