from ...core.registry import TaskSpec

from .config import CoalitionsDataConfig
from .data import build_dataloaders, prepare_coalitions
from .models import MODELS
from .callbacks import coalitions_callbacks
from .loss import build_coalitions_loss_fn

TASK = TaskSpec(
    name='coalitions',
    data_config=CoalitionsDataConfig,
    dataloader_builder=build_dataloaders, # type:ignore
    prepare=prepare_coalitions,
    models=MODELS,
    callbacks=coalitions_callbacks,
    loss_builder=build_coalitions_loss_fn,
)

__all__ = [
    'TASK',
    'CoalitionsDataConfig',
]
