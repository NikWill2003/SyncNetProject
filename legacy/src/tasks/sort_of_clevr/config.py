from __future__ import annotations

from dataclasses import dataclass

from ...core.config import DataConfig


@dataclass
class SortOfClevrDataConfig(DataConfig):
    name: str = 'sort_of_clevr'
    seed: int = 1
    root: str = './data'
    dir: str = 'sort-of-clevr'
    train_size: int = 9800
    test_size: int = 200
    img_size: int = 75
    obj_size: int = 5
    nb_questions: int = 10
    t_subtype: int = -1
