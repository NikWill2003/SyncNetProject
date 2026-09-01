from dataclasses import dataclass


from ...core.config import DataConfig

# constants for sort-of-clevr dataset

QUESTION_SIZE: int = 18  # 2 x 6 colour one-hot, 3 question type, 3 subtype
ANSWER_SIZE: int = 10  # yes, no, rectangle, circle, r, g, b, o, k, y
Q_TYPE_IDX: int = 12
SUB_Q_TYPE_IDX: int = 15

IMG_SIZE = 75
OBJ_SIZE = 5
NB_QUESTIONS = 10
T_SUBTYPE = -1

Q_TYPES_OFFSET = {
    'non_relational': 0,
    'binary': 1,
    'ternary': 2,
}

COLOURS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'orange': (0, 156, 255),
    'grey': (128, 128, 128),
    'yellow': (0, 255, 255),
}

ANSWERS = [
    'yes',
    'no',
    'rectangle',
    'circle',
    'red',
    'green',
    'blue',
    'orange',
    'grey',
    'yellow',
]

# index at which the count answers start
COUNT_OFFSET: int = 4

COUNT_ANSWERS = [
    'yes',
    'no',
    'rectangle',
    'circle',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
]

SUBTYPE_NAMES = {
    ('non_relational', 0): 'query_shape',
    ('non_relational', 1): 'left_of_centre',
    ('non_relational', 2): 'top_half',
    ('binary', 0): 'closest_shape',
    ('binary', 1): 'furthest_shape',
    ('binary', 2): 'count_same_shape',
    ('ternary', 0): 'count_in_box',
    ('ternary', 1): 'on_band',
    ('ternary', 2): 'count_obtuse',
}

# config for soc

@dataclass
class SortOfClevrDataConfig(DataConfig):
    name: str = 'sort_of_clevr'
    seed: int = 1
    root: str = './data'
    dir: str = 'sort-of-clevr'
    train_size: int = 9800
    test_size: int = 200
    nb_questions: int = NB_QUESTIONS
    t_subtype: int = T_SUBTYPE


NAME = 'sort_of_clevr'
DATA_CONFIG = SortOfClevrDataConfig
ANSWER_DIM = ANSWER_SIZE
N_COLOURS = len(COLOURS)


