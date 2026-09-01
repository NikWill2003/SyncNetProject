import torch.nn as nn
from torch import Tensor

from ...datasets.soc import spec as SOCConst
from ...datasets.sqoop import spec as SQOOPConst


class IdentityQuestionEncoder(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()

        self.output_shape = (input_dim,)

    def forward(self, question: Tensor) -> Tensor:
        return question.float()


class MLPQuestionEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.output_shape = (output_dim,)

    def forward(self, question: Tensor) -> Tensor:
        return self.mlp(question.float())


class TokeniserQuestionEncoder(nn.Module):

    def __init__(self, vocab_size: int, question_len: int, embedding_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_shape = (question_len, embedding_dim)

    def forward(self, question: Tensor) -> Tensor:
        return self.embedding(question)


class LSTMQuestionEncoder(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, output_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, output_dim, batch_first=True)

        self.output_shape = (output_dim,)

    def forward(self, question: Tensor) -> Tensor:
        x = self.embedding(question)
        _, (h, _) = self.lstm(x)

        return h[-1]


QuestionEncoder = IdentityQuestionEncoder | MLPQuestionEncoder | TokeniserQuestionEncoder | LSTMQuestionEncoder


def build_question_encoder(enc_config: dict, dataset: str, allowed: set[str] | None = None) -> QuestionEncoder:
    name = enc_config.get('name')

    if allowed is not None and name not in allowed:
        raise ValueError(f'question encoder {name!r} is not allowed here; choose from {sorted(allowed)}')

    if dataset == 'sort_of_clevr':
        input_dim = SOCConst.QUESTION_SIZE
        vocab_size = None
    elif dataset == 'sqoop':
        input_dim = SQOOPConst.QUESTION_SIZE
        vocab_size = SQOOPConst.VOCAB_SIZE
    else:
        raise ValueError(f'unrecognised dataset: {dataset!r}')

    if name in ('tokenise', 'lstm') and vocab_size is None:
        raise ValueError(
            f'{name!r} needs token indices, but {dataset!r} questions are a feature vector; use identity or mlp'
        )

    if name == 'identity':
        return IdentityQuestionEncoder(input_dim)

    if name == 'mlp':
        return MLPQuestionEncoder(
            input_dim=input_dim,
            hidden_dim=enc_config['hidden_dim'],
            output_dim=enc_config['output_dim'],
        )

    if name == 'tokenise':
        assert vocab_size is not None
        return TokeniserQuestionEncoder(
            vocab_size=vocab_size,
            question_len=input_dim,
            embedding_dim=enc_config['embedding_dim'],
        )

    if name == 'lstm':
        assert vocab_size is not None
        return LSTMQuestionEncoder(
            vocab_size=vocab_size,
            embedding_dim=enc_config['embedding_dim'],
            output_dim=enc_config['output_dim'],
        )

    raise ValueError(f'unrecognised question encoder {name!r} for dataset {dataset!r}')