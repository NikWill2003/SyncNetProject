from __future__ import annotations

from dataclasses import dataclass

from ...core.config import DataConfig


@dataclass
class SqoopDataConfig(DataConfig):
    name: str = 'sqoop'
    seed: int = 1
    root: str = './data'
    dir: str = 'sqoop-rhs18-n1080000'

    train_size: int = 1_080_000 # total examples
    test_size: int = 25_600 # total examples

    # systematic split: rhs per lhs seen in train (1..35)
    rhs_variety: int = 18

    # which on-disk splits fill the trainer's eval / test slots.
    # 'val_unseen' exists on disk for diagnostics / a future 4-loader
    # trainer. Early stopping must use val_seen.
    eval_split: str = 'val_seen'
    test_split: str = 'test_unseen'

    # scene rendering
    img_size: int = 64
    num_objects: int = 5
    min_obj_size: int = 10
    max_obj_size: int = 15

    