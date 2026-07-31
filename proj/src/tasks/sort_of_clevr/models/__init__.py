from ....core.registry import ModelSpec

from .transformer import SortOfClevrTransformer, SortOfClevrTransformerConfig
from .syncnet_v1 import SortOfClevrSyncNetV1, SortOfClevrSyncNetV1Config
from .syncnet_v2 import SortOfClevrSyncNetV2, SortOfClevrSyncNetV2Config
from .syncnet_v3 import SortOfClevrSyncNetV3, SortOfClevrSyncNetV3Config
from .recurrent_syncnet import (
    SortOfClevrRecurrentSyncNet,
    SortOfClevrRecurrentSyncNetConfig,
)

MODELS: dict[str, ModelSpec] = {
    'sort_of_clevr_transformer': ModelSpec(
        config=SortOfClevrTransformerConfig,
        model_class=SortOfClevrTransformer,
    ),
    'sort_of_clevr_syncnet_v1': ModelSpec(
        config=SortOfClevrSyncNetV1Config,
        model_class=SortOfClevrSyncNetV1,
    ),
    'sort_of_clevr_syncnet_v2': ModelSpec(
        config=SortOfClevrSyncNetV2Config,
        model_class=SortOfClevrSyncNetV2,
    ),
    'sort_of_clevr_syncnet_v3': ModelSpec(
        config=SortOfClevrSyncNetV3Config,
        model_class=SortOfClevrSyncNetV3,
    ),
    'sort_of_clevr_recurrent_syncnet': ModelSpec(
        config=SortOfClevrRecurrentSyncNetConfig,
        model_class=SortOfClevrRecurrentSyncNet,
    ),
}

__all__ = [
    'MODELS',
    'SortOfClevrTransformer',
    'SortOfClevrSyncNetV1',
    'SortOfClevrSyncNetV2',
    'SortOfClevrSyncNetV3',
    'SortOfClevrRecurrentSyncNet',
    'SortOfClevrTransformerConfig',
    'SortOfClevrSyncNetV1Config',
    'SortOfClevrSyncNetV2Config',
    'SortOfClevrSyncNetV3Config',
    'SortOfClevrRecurrentSyncNetConfig',
]
