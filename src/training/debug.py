from typing import Any, Iterator

from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter

from .logging import (
    cmdline_format_metrics, section, MultiAverageMeter
)

from .utils import (
    batch_iter,
    get_batch_dict_size,
)

from ..core import Config


class DebugTrainer:

    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        logger: MultiProcessAdapter,
        dataloaders: tuple[DataLoader | Iterator[dict[str, Tensor]], ...],
        forward_args: dict[str, Any],
        optimiser: Optimizer,
        scheduler: LRScheduler,
        accelerator: Accelerator,
        loss_fn,
        ) -> None:

        self.cfg = cfg
        self.train_cfg = cfg.train

        self.logger = logger
        self.model = model
        self.forward_args = forward_args
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.loss_fn = loss_fn
        
    
        self.train_dataloader, _, _ = dataloaders

        # prepare model, optimiser, lr_scheduler and optionally dataloaders unless cached
        if isinstance(self.train_dataloader, DataLoader):
                (
                self.model, 
                self.optimiser, 
                self.scheduler,
                self.train_dataloader,
            ) = self.accelerator.prepare(
                self.model, 
                self.optimiser, 
                self.scheduler,
                self.train_dataloader,
            )
        else:    
            (
                self.model, 
                self.optimiser, 
                self.scheduler,
            ) = self.accelerator.prepare(
                self.model, 
                self.optimiser, 
                self.scheduler,
            )

        self.train_meter = MultiAverageMeter()
        
        # make dataloader an infinite step iterator
        self.train_batch_iter = batch_iter(self.train_dataloader)

        #step tracking
        self.total_step = 0
        self.opt_step = 0

    #step/interval tracking:

    def step(self) -> None:
        self.total_step += 1
        self.opt_step = self.total_step // self.train_cfg.grad_accum

    def hit_step_interval(self, interval: int) -> bool:
                
        if not (self.opt_step % interval == 0):
            return False

        if not (self.total_step % self.train_cfg.grad_accum == 0):
            return False
        
        return True
    
    # train and eval functions:

    def train_step(
        self,
        batch: dict[str, Any],
        forward_args: dict[str, Any],
        max_grad_norm: float=1.0
        ) -> dict[str, float]:

        self.model.train()
        metrics = {}

        out: dict = self.model(**batch, **forward_args)
        loss, loss_metrics = self.loss_fn(out, batch)
        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients and max_grad_norm:
            grad_norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), max_grad_norm
                )

            if isinstance(grad_norm, Tensor):
                metrics['grad_norm'] = grad_norm.item()

        self.optimiser.step()
        self.scheduler.step()
        self.optimiser.zero_grad()

        metrics |= section(out.get('metrics', {}), 'model')
        metrics |= section(loss_metrics, 'loss')

        return metrics
    
    def debug_model(self):
        """
        Train the model on one batch to check if it converges
        """

        batch = next(self.train_batch_iter)

        while self.opt_step < self.train_cfg.n_steps:
            self.step()
            
            with self.accelerator.accumulate(self.model):
                train_step_metrics = self.train_step(
                    batch, self.forward_args, self.train_cfg.grad_clip
                )

            self.train_meter.update(train_step_metrics, n=get_batch_dict_size(batch))

            if self.hit_step_interval(self.cfg.logging.train_log_interval):
                train_metrics = self.train_meter.get_averages()
                self.logger.info(
                    '[ %-5s | Step: %-6d | %s ]', 
                    "train".capitalize(), 
                    self.opt_step, 
                    cmdline_format_metrics(train_metrics)
                    )
                
                self.train_meter.reset()
