from ....core.registry import ModelSpec

from .conv_lstm import SqoopConvLSTM, SqoopConvLSTMConfig
from .question_only import SqoopQuestionOnly, SqoopQuestionOnlyConfig
from .syncnet import SqoopSyncNet, SqoopSyncNetConfig
from .transformer import SqoopTransformer, SqoopTransformerConfig

MODELS: dict[str, ModelSpec] = {
    'sqoop_question_only': ModelSpec(
        config=SqoopQuestionOnlyConfig,
        model_class=SqoopQuestionOnly,
    ),
    'sqoop_conv_lstm': ModelSpec(
        config=SqoopConvLSTMConfig,
        model_class=SqoopConvLSTM,
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
