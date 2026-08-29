from ....core.registry import ModelSpec

from .busnet import SqoopBusNet, SqoopBusNetConfig
from .conv_lstm import SqoopConvLSTM, SqoopConvLSTMConfig
from .film import SqoopFiLM, SqoopFiLMConfig
from .relnet import SqoopRelNet, SqoopRelNetConfig
from .question_only import SqoopQuestionOnly, SqoopQuestionOnlyConfig
from .syncnet import SqoopSyncNet, SqoopSyncNetConfig
from .transformer import SqoopTransformer, SqoopTransformerConfig

MODELS: dict[str, ModelSpec] = {
    'sqoop_busnet': ModelSpec(
        config=SqoopBusNetConfig,
        model_class=SqoopBusNet,
    ),
    'sqoop_question_only': ModelSpec(
        config=SqoopQuestionOnlyConfig,
        model_class=SqoopQuestionOnly,
    ),
    'sqoop_conv_lstm': ModelSpec(
        config=SqoopConvLSTMConfig,
        model_class=SqoopConvLSTM,
    ),
    'sqoop_film': ModelSpec(
        config=SqoopFiLMConfig,
        model_class=SqoopFiLM,
    ),
    'sqoop_relnet': ModelSpec(
        config=SqoopRelNetConfig,
        model_class=SqoopRelNet,
    ),
    'sqoop_transformer': ModelSpec(
        config=SqoopTransformerConfig,
        model_class=SqoopTransformer,
    ),
    'sqoop_syncnet': ModelSpec(
        config=SqoopSyncNetConfig,
        model_class=SqoopSyncNet,
    ),
}

__all__ = ['MODELS']
