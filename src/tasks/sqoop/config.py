from __future__ import annotations

from dataclasses import dataclass

from ...core.config import DataConfig


@dataclass
class SqoopDataConfig(DataConfig):
    name: str = 'sqoop'
    seed: int = 1
    root: str = './data'
    dir: str = 'sqoop-rhs18'

    # systematic split
    rhs_variety: int = 18          # rhs per lhs seen in train (1..35)
    num_repeats: int = 80          # train examples per seen pair
    num_repeats_eval: int = 10     # eval examples per pair
    max_train_pairs: int = 0       # 0 = keep all generated train examples

    # which on-disk splits fill the trainer's eval / test slots.
    # 'val_unseen' exists on disk for diagnostics / a future 4-loader
    # trainer. Early stopping must use val_seen.
    eval_split: str = 'val_seen'
    test_split: str = 'test_unseen'

    # 0 = full train split; >0 = deterministic random subsample (GPU
    # cache fitting / data-efficiency runs)
    max_train_examples: int = 0

    # scene rendering
    img_size: int = 64
    num_objects: int = 5
    min_obj_size: int = 10
    max_obj_size: int = 15
