"""Task-agnostic VQA models.

Everything here takes a `QuestionEncoder` and plain ints (img_size,
answer_dim) -- no task config, no task contracts. A task package supplies
the question encoder and a thin adapter (see `src/models/adapters.py`)
and gets the whole family for free.
"""

from .adapters import VQAAdapter, ImageQuestionAdapter, QuestionOnlyAdapter
from .encoders import (
    PatchifyEncoder, CNNEncoder, EncoderConfig,
    PatchifyEncoderConfig, CNNEncoderConfig,
)
from .pos_enc import PositionalEncoder1D, PositionalEncoder2D
from .question_encoders import (
    QuestionEncoder, IdentityQuestionEncoder, TokenEmbedQuestionEncoder,
)
from .film import VQAFiLM, VQAFiLMConfig
from .question_only import VQAQuestionOnly
from .relnet import VQARelNet, VQARelNetConfig
from .syncnet import VQASyncNet, VQASyncNetConfig
from .phasebind import PhaseBind, PhaseBindConfig
from .osc_field import OscField, OscFieldConfig
from .object_tokens import ObjectTokenizer
from .vqa_transformer import VQATransformer

__all__ = [
    'VQAAdapter', 'ImageQuestionAdapter', 'QuestionOnlyAdapter',
    'PatchifyEncoder', 'CNNEncoder', 'EncoderConfig',
    'PatchifyEncoderConfig', 'CNNEncoderConfig',
    'PositionalEncoder1D', 'PositionalEncoder2D',
    'QuestionEncoder', 'IdentityQuestionEncoder',
    'TokenEmbedQuestionEncoder',
    'VQAFiLM', 'VQAFiLMConfig', 'VQARelNet', 'VQARelNetConfig',
    'VQAQuestionOnly', 'VQASyncNet', 'VQASyncNetConfig', 'VQATransformer',
    'PhaseBind', 'PhaseBindConfig', 'OscField', 'OscFieldConfig', 'ObjectTokenizer',
]
