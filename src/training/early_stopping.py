from typing import Any, Optional
import math

import torch
import torch.nn as nn
from torch import Tensor
from accelerate import Accelerator

class EarlyStoppingManager:
    def __init__(
        self,
        model: nn.Module,
        val_metric: str,
        big_is_better: bool,
        patience: Optional[int] = 10,
        min_delta: float = 0.0,
        accelerator: Optional[Accelerator] = None
    ) -> None:
        
        self._model = model
        self._metric = val_metric
        self._big_is_better = big_is_better
        self._patience = patience
        self._min_delta = min_delta
        self._accelerator = accelerator

        self._best_val = float('-inf') if big_is_better else float('inf')
        self._best_step = 0
        self._best_state = self.get_state(model)

        self._num_bad_steps = 0
        self._should_stop = False

    
    def get_state(self, model: nn.Module) -> dict[str, Tensor]:
        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()
            return self._accelerator.get_state_dict(model) # type: ignore
        
        return {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    def check_should_stop(self) -> bool:
        
        if self._patience is None:
            return False
        return self._num_bad_steps >= self._patience

    def is_improvement(self, cur: float) -> bool:
        if self._big_is_better:
            return cur > self._best_val + self._min_delta
        else:
            return cur < self._best_val - self._min_delta

    def update(self, step_metrics: dict[str, float], step: int) -> dict[str, Any]:
        step_val = step_metrics.get(self._metric)

        if step_val is None:
            raise ValueError(f'{self._metric} missing in step metrics: {step_metrics}')

        step_val = float(step_val)

        if not math.isfinite(step_val):
            self._num_bad_steps += 1
            self._should_stop = self.check_should_stop()
            return self.summary()

        if self.is_improvement(step_val):
            self._best_val = step_val
            self._best_step = step
            self._best_state = self.get_state(self._model)
            self._num_bad_steps = 0
        else:
            self._num_bad_steps += 1

        self._should_stop = self.check_should_stop()

        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            f'best_{self._metric}': self._best_val,
            'best_step': self._best_step,
            'num_bad_steps': self._num_bad_steps,
            'should_stop': self._should_stop,
        }
    
    def get_best_stats(self) -> dict[str, float | int]:
        return {
            f'best_{self._metric}': self._best_val,
            'best_step': self._best_step,
        }

    def get_best_state(self) -> dict[str, Tensor]:
            
        if self._best_state is None:
            raise RuntimeError('No best state has been saved yet.')

        return self._best_state

    def load_best_model(self) -> None:

        if self._best_state is None:
            raise RuntimeError('No best state has been saved yet.')
        
        if self._accelerator is not None:
            (self._accelerator.unwrap_model(self._model)
             .load_state_dict(self._best_state))
            
            return

        self._model.load_state_dict(self._best_state)

    def save_best_model(self, out_dir: str, fname: str='best_model.pt') -> str:

        if self._best_state is None:
            raise RuntimeError('No best state to be saved yet.')

        from pathlib import Path
        path = Path(out_dir) / fname

        torch.save(self._best_state, path)

        return str(path)
