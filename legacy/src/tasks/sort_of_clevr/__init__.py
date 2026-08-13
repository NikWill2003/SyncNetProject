from ...core.registry import TaskSpec

from .config import SortOfClevrDataConfig
from .data import (
    build_dataloaders,
    prepare_sort_of_clevr,
    translate_question,
    translate_answer,
)
from .models import MODELS
from .callbacks import sort_of_clevr_callbacks
from .loss import build_cross_entropy

TASK = TaskSpec(
    name='sort_of_clevr',
    data_config=SortOfClevrDataConfig,
    dataloader_builder=build_dataloaders, #type: ignore
    prepare=prepare_sort_of_clevr,
    models=MODELS,
    callbacks=sort_of_clevr_callbacks,
    loss_builder=build_cross_entropy,
)

__all__ = [
    'TASK',
    'SortOfClevrDataConfig',
    'translate_question',
    'translate_answer',
]
