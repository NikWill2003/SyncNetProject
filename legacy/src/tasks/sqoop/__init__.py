from ...core.registry import TaskSpec

from .config import SqoopDataConfig
from .data import build_dataloaders, prepare_sqoop, decode_question
from .models import MODELS
from .callbacks import sqoop_callbacks
from .loss import build_cross_entropy

TASK = TaskSpec(
    name='sqoop',
    data_config=SqoopDataConfig,
    dataloader_builder=build_dataloaders,  # type: ignore
    prepare=prepare_sqoop,
    models=MODELS,
    callbacks=sqoop_callbacks,
    loss_builder=build_cross_entropy,
)

__all__ = ['TASK', 'SqoopDataConfig', 'decode_question']
