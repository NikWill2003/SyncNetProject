from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ...training import Trainer
    from ..config import Config, CallbackConfig


class BaseCallBack:
    """Hooks receive the model output and the batch as two separate dicts
    (never a merged splat): provenance stays explicit and output/batch key
    collisions are impossible. See tasks/<task>/contracts.py for the keys
    each dict carries.

    A model must run correctly on the batch alone -- `model(**batch)`.
    Static forward behaviour is a field on the model config; a callback
    that needs a different forward (traces, `t_override`, an ablation)
    calls the model itself, which also keeps the cost on the few batches
    it actually samples rather than on every step.

    Every hook runs under `torch.no_grad()` (see CallBackList): callbacks
    are measurement, and `out` still carries a live graph at
    on_train_step_end, so an unguarded callback that keeps a reference to
    a tensor pins the whole graph. A callback that genuinely needs grad
    can re-enable it locally with `with torch.enable_grad():`.
    """

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
