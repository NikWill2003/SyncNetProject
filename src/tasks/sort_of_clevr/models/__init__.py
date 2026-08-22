from ....core.registry import ModelSpec

from .transformer import SortOfClevrTransformer, SortOfClevrTransformerConfig
from .question_only import (
    SortOfClevrQuestionOnly, SortOfClevrQuestionOnlyConfig,
)
from .syncnet import SortOfClevrSyncNet, SortOfClevrSyncNetConfig
from .conv_lstm import SortOfClevrCNNMLP, SortOfClevrCNNMLPConfig
from .film import SortOfClevrFiLM, SortOfClevrFiLMConfig
from .relnet import SortOfClevrRelNet, SortOfClevrRelNetConfig

MODELS: dict[str, ModelSpec] = {
    'sort_of_clevr_transformer': ModelSpec(
        config=SortOfClevrTransformerConfig,
        model_class=SortOfClevrTransformer,
    ),
    'sort_of_clevr_question_only': ModelSpec(
        config=SortOfClevrQuestionOnlyConfig,
        model_class=SortOfClevrQuestionOnly,
    ),
    'sort_of_clevr_cnn_mlp': ModelSpec(
        config=SortOfClevrCNNMLPConfig,
        model_class=SortOfClevrCNNMLP,
    ),
    'sort_of_clevr_film': ModelSpec(
        config=SortOfClevrFiLMConfig,
        model_class=SortOfClevrFiLM,
    ),
    'sort_of_clevr_relnet': ModelSpec(
        config=SortOfClevrRelNetConfig,
        model_class=SortOfClevrRelNet,
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
    'SortOfClevrCNNMLP', 'SortOfClevrCNNMLPConfig',
    'SortOfClevrFiLM', 'SortOfClevrFiLMConfig',
    'SortOfClevrRelNet', 'SortOfClevrRelNetConfig',
]
