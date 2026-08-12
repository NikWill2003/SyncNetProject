from ....core.registry import ModelSpec

from .transformer import SortOfClevrTransformer, SortOfClevrTransformerConfig
from .question_only import (
    SortOfClevrQuestionOnly, SortOfClevrQuestionOnlyConfig,
)
from .syncnet import SortOfClevrSyncNet, SortOfClevrSyncNetConfig

MODELS: dict[str, ModelSpec] = {
    'sort_of_clevr_transformer': ModelSpec(
        config=SortOfClevrTransformerConfig,
        model_class=SortOfClevrTransformer,
    ),
    'sort_of_clevr_question_only': ModelSpec(
        config=SortOfClevrQuestionOnlyConfig,
        model_class=SortOfClevrQuestionOnly,
    ),
    'sort_of_clevr_syncnet': ModelSpec(
        config=SortOfClevrSyncNetConfig,
        model_class=SortOfClevrSyncNet,
    ),
}

# superseded models live in .legacy and are deliberately NOT registered
__all__ = [
    'MODELS',
    'SortOfClevrTransformer',
    'SortOfClevrTransformerConfig',
    'SortOfClevrQuestionOnly',
    'SortOfClevrQuestionOnlyConfig',
    'SortOfClevrSyncNet',
    'SortOfClevrSyncNetConfig',
]
