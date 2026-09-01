import string
from dataclasses import dataclass


from ...core.config import DataConfig

# sqoop constants:

# uppercase chars + digits
SHAPES: list[str] = list(string.ascii_uppercase) + [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'
]
RELATIONS: list[str] = ['left_of', 'right_of', 'above', 'below']

N_SHAPES = len(SHAPES) # 36
N_RELATIONS = len(RELATIONS) # 4
VOCAB_SIZE = N_SHAPES + N_RELATIONS  # 40

SHAPE_TO_IDX = {s: i for i, s in enumerate(SHAPES)}
REL_TO_IDX = {r: N_SHAPES + i for i, r in enumerate(RELATIONS)}
IDX_TO_TOKEN = SHAPES + RELATIONS

QUESTION_SIZE = 3
ANSWER_SIZE = 2 # true or false

IMG_SIZE = 64
NUM_OBJECTS = 5
MIN_OBJECT_SIZE = 10
MAX_OBJECT_SIZE = 15


@dataclass
class SqoopDataConfig(DataConfig):
    name: str = 'sqoop'
    seed: int = 1
    root: str = './data'
    dir: str = 'sqoop-rhs18-n1080000'

    train_size: int = 1_080_000
    test_size: int = 25_600

    # systematic split: rhs per lhs seen in train (1, 2, 4, 8, 16, 35)
    rhs_variety: int = 18




NAME = 'sqoop'
DATA_CONFIG = SqoopDataConfig
ANSWER_DIM = ANSWER_SIZE


