from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass

import torch

if TYPE_CHECKING:
    from ..training import Trainer
    from .config import Config, CallbackConfig


class BaseCallBack:

    def on_train_step_end(
            self, trainer: Trainer, out: dict, batch: dict
            ) -> Optional[dict[str, float]]:
        pass

    def on_eval_step_end(
            self, trainer: Trainer, out: dict, batch: dict
            ) -> Optional[dict[str, float]]:
        pass

    def on_test_step_end(
            self, trainer: Trainer, out: dict, batch: dict
            ) -> Optional[dict[str, float]]:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    @classmethod
    def from_config(cls, cfg: Config, cb_cfg: CallbackConfig) -> BaseCallBack:
        return cls()


class CallBackList:

    def __init__(self, callbacks: list[BaseCallBack]) -> None:
        self.callbacks = callbacks

    @torch.no_grad()
    def on_train_step_end(
            self, trainer: Trainer, out: dict, batch: dict
            ) -> dict[str, float]:

        metrics = {}
        for callback in self.callbacks:
            metrics |= callback.on_train_step_end(trainer, out, batch) or {}
        return metrics

    @torch.no_grad()
    def on_eval_step_end(
            self, trainer: Trainer, out: dict, batch: dict
            ) -> dict[str, float]:

        metrics = {}
        for callback in self.callbacks:
            metrics |= callback.on_eval_step_end(trainer, out, batch) or {}
        return metrics

    @torch.no_grad()
    def on_test_step_end(
            self, trainer: Trainer, out: dict, batch: dict
            ) -> dict[str, float]:

        metrics = {}
        for callback in self.callbacks:
            metrics |= callback.on_test_step_end(trainer, out, batch) or {}
        return metrics

    @torch.no_grad()
    def on_train_end(self, trainer: Trainer) -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)


def accuracy(logits, targets) -> float:
    """Top-1 accuracy, nan when the logits are nan or the group is empty.

    The nan guard is deliberate: argmax over nan logits returns an index
    regardless, so without it a diverged run reports a plausible-looking
    number instead of an obvious one.
    """
    if logits.numel() == 0 or torch.isnan(logits).any():
        return float('nan')
    return float((logits.argmax(-1) == targets).float().mean())


@dataclass(frozen=True)
class CallbackSpec:
    config: type[CallbackConfig]
    callback_class: type[BaseCallBack]
    requires: frozenset[str] = frozenset()

# Callbacks that need only the batch/output contract would live here.
# There are none: every metric so far depends on a dataset's question
# layout, so they live in datasets/<name>/callbacks.py.
SHARED_CALLBACKS: dict[str, CallbackSpec] = {}