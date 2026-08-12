from ....core.registry import ModelSpec

from .adapters import (
    SqoopSyncNet, SqoopSyncNetConfig,
    SqoopConvLSTM, SqoopConvLSTMConfig,
    SqoopQuestionOnly, SqoopQuestionOnlyConfig,
)

MODELS: dict[str, ModelSpec] = {
    'sqoop_syncnet': ModelSpec(
        config=SqoopSyncNetConfig,
        model_class=SqoopSyncNet,
    ),
    'sqoop_question_only': ModelSpec(
        config=SqoopQuestionOnlyConfig,
        model_class=SqoopQuestionOnly,
    ),
    'sqoop_conv_lstm': ModelSpec(
        config=SqoopConvLSTMConfig,
        model_class=SqoopConvLSTM,
    ),
}

__all__ = ['MODELS']
