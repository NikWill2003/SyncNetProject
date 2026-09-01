from __future__ import annotations

import itertools
import time

from .logging import MultiAverageMeter
from .trainer import Trainer, cuda_sync
from ..utils import get_batch_dict_size


class DebugTrainer(Trainer):
    """Train on one batch, for `train.n_steps` steps. No eval, no test.

    Everything else is the normal Trainer: the same train_step, the same
    optimiser, the same callbacks and logging. That is the point -- a
    parallel implementation would drift from the real path and pass while
    the real path was broken.

    `Trainer.train()` pulls from `self.train_batch_iter`, so repeating one
    batch is the whole trick and the loop is inherited unchanged.
    """

    def on_train_start(self) -> None:
        self.train_meter = MultiAverageMeter()

        # taken once and held: taken here rather than in __init__ because
        # the dataloader may have been through accelerator.prepare() by
        # then, and once because the on-device loaders reshuffle on every
        # __iter__, so re-deriving it would give a different batch
        self.batch = next(iter(self.train_dataloader))
        self.batch_size = get_batch_dict_size(self.batch)
        self.train_batch_iter = itertools.repeat(self.batch)

        self.total_step = 0
        self.opt_step = 0
        self.should_stop = False

        self.train_init_log()
        self.log_info(
            f'DEBUG | one batch of {self.batch_size} | '
            f'{self.train_cfg.n_steps} steps | no eval, no test'
        )

        cuda_sync()
        self.step_interval_start = time.perf_counter()
        self.tot_training_time = 0

    def on_eval_hit(self) -> None:
        pass

    def on_train_end(self) -> dict[str, float]:
        averages = self.train_meter.get_averages()
        return {'loss': float(averages.get('loss', float('nan')))}
