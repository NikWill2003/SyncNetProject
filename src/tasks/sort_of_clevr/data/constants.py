QUESTION_SIZE: int = 18  # 2 x 6 colour one-hot, 3 question type, 3 subtype
Q_TYPE_IDX: int = 12
SUB_Q_TYPE_IDX: int = 15
ANSWER_SIZE: int = 10  # yes, no, rectangle, circle, r, g, b, o, k, y

Q_TYPES_OFFSET: dict[str, int] = {
    'non_relational': 0,
    'binary': 1,
    'ternary': 2,
}

COLOURS: dict[str, tuple[int, int, int]] = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'orange': (0, 156, 255),
    'grey': (128, 128, 128),
    'yellow': (0, 255, 255),
}

ANSWERS: list[str] = [
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

COUNT_ANSWERS: list[str] = [
    'yes',
    'no',
    'rectangle',
    'circle',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
]