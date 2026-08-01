from ....core.registry import ModelSpec

from .adapters import (
    SqoopRecurrentSyncNet, SqoopRecurrentSyncNetConfig,
    SqoopSyncNetV3, SqoopSyncNetV3Config,
    SqoopSyncNetV1, SqoopSyncNetV1Config,
    SqoopConvLSTM, SqoopConvLSTMConfig,
)

MODELS: dict[str, ModelSpec] = {
    'sqoop_recurrent_syncnet': ModelSpec(
        config=SqoopRecurrentSyncNetConfig,
        model_class=SqoopRecurrentSyncNet,
    ),
    'sqoop_syncnet_v3': ModelSpec(
        config=SqoopSyncNetV3Config,
        model_class=SqoopSyncNetV3,
    ),
    'sqoop_syncnet_v1': ModelSpec(
        config=SqoopSyncNetV1Config,
        model_class=SqoopSyncNetV1,
    ),
    'sqoop_conv_lstm': ModelSpec(
        config=SqoopConvLSTMConfig,
        model_class=SqoopConvLSTM,
    ),
}

__all__ = ['MODELS']
