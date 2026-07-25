from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...training import Trainer
    from ..config import Config, CallbackConfig


class BaseCallBack:
    """Hooks receive the model output and the batch as two separate dicts
    (never a merged splat): provenance stays explicit and output/batch key
    collisions are impossible. See tasks/<task>/contracts.py for the keys
    each dict carries.
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

    def on_train_step_end(
            self, trainer: Trainer, out: dict, batch: dict
            ) -> dict[str, float]:

        metrics = {}
        for callback in self.callbacks:
            metrics |= callback.on_train_step_end(trainer, out, batch) or {}
        return metrics

    def on_eval_step_end(
            self, trainer: Trainer, out: dict, batch: dict
            ) -> dict[str, float]:

        metrics = {}
        for callback in self.callbacks:
            metrics |= callback.on_eval_step_end(trainer, out, batch) or {}
        return metrics

    def on_test_step_end(
            self, trainer: Trainer, out: dict, batch: dict
            ) -> dict[str, float]:

        metrics = {}
        for callback in self.callbacks:
            metrics |= callback.on_test_step_end(trainer, out, batch) or {}
        return metrics

    def on_train_end(self, trainer: Trainer) -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)
