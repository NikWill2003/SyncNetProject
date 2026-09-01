"""Model registry.

    common/       img_enc, pos_enc, qst_enc -- built from config by
                  build_image_encoder / build_question_encoder
    baseline/     the published comparison set and the floors
    sync/         the architectures under test

Every model takes the batch (core/contracts.VQABatch) and returns
{'logits': ...}, and is built by
`Model.from_config(model_cfg, data_cfg, dataset)`.
"""

from .baseline.conv import VQAConv, VQAConvConfig
from .baseline.film import VQAFiLM, VQAFiLMConfig
from .baseline.question_only import VQAQuestionOnly, VQAQuestionOnlyConfig
from .baseline.relnet import VQARelNet, VQARelNetConfig
from .baseline.shared_workspace import (
    VQAWorkspaceTransformer, VQAWorkspaceTransformerConfig,
)
from .baseline.transformer import VQATransformer, VQATransformerConfig
from .sync.busnet import BusNet, BusNetConfig
from .sync.gated import GatedNet, GatedNetConfig
from .sync.identity_busnet import IdentityBusNet, IdentityBusNetConfig
from .sync.token_busnet import TokenBusNet, TokenBusNetConfig

MODELS: dict[str, tuple[type, type]] = {
    'busnet':           (BusNetConfig,          BusNet),
    'identity_busnet':  (IdentityBusNetConfig,  IdentityBusNet),
    'token_busnet':     (TokenBusNetConfig,     TokenBusNet),
    'gated':            (GatedNetConfig,        GatedNet),
    'transformer':      (VQATransformerConfig,  VQATransformer),
    'shared_workspace': (
        VQAWorkspaceTransformerConfig, VQAWorkspaceTransformer),
    'relnet':           (VQARelNetConfig,       VQARelNet),
    'film':             (VQAFiLMConfig,         VQAFiLM),
    'conv':             (VQAConvConfig,         VQAConv),
    'question_only':    (VQAQuestionOnlyConfig, VQAQuestionOnly),
}
