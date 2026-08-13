"""SQOOP callbacks: accuracy (overall + per-relation) plus the shared
t_variance ablation. t_variance is imported from sort_of_clevr rather
than duplicated -- it only touches batch keys (images / questions /
answers) and the model's t_override interface, both shared by the sqoop
contracts."""

from ....core.registry import CallbackSpec
from .metrics import sqoop_metric_callbacks
from ...sort_of_clevr.callbacks.t_variance import (
    sort_of_clevr_t_variance_callbacks,
)

sqoop_callbacks: dict[str, CallbackSpec] = (
    sqoop_metric_callbacks
    | sort_of_clevr_t_variance_callbacks
)

__all__ = ['sqoop_callbacks']
