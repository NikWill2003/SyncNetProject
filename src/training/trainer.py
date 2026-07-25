from typing import Any, Literal, Iterator

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter

from .early_stopping import EarlyStoppingManager
from .logging import (
    wandb_format_metrics,
    cmdline_format_metrics,
    section,
    desection,
    MultiAverageMeter,
)

from .utils import (
    batch_iter,
    get_batch_dict_size,
    get_param_count,
)
from ..core.callbacks import CallBackList

from ..core import Config


class Trainer:

    def __init__(
        self,
        cfg: Config,
        out_dir: str,
        logger: MultiProcessAdapter,
        model: nn.Module,
        dataloaders: tuple[DataLoader | Iterator[dict[str, Tensor]], ...],
        forward_args: dict[str, Any],
        optimiser: Optimizer,
        scheduler: LRScheduler,
        accelerator: Accelerator,
        callbacks: CallBackList,
        loss_fn,
        ) -> None:

        self.cfg = cfg
        self.train_cfg = cfg.train

        self.out_dir = out_dir
        self.logger = logger
        self.model = model
        self.forward_args = forward_args
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.loss_fn = loss_fn
        self.callbacks = callbacks
        (
            self.train_dataloader,
            self.eval_dataloader,
            self.test_dataloader
        ) = dataloaders

        # prepare model, optimiser, lr_scheduler and optionally dataloaders unless cached
        if isinstance(self.train_dataloader, DataLoader):
                (
                self.model, 
                self.optimiser, 
                self.scheduler,
                self.train_dataloader,
                self.eval_dataloader,
                self.test_dataloader
            ) = self.accelerator.prepare(
                self.model, 
                self.optimiser, 
                self.scheduler,
                self.train_dataloader,
                self.eval_dataloader,
                self.test_dataloader
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

        self.early_stopping = EarlyStoppingManager(
            self.model, 
            self.train_cfg.early_stop_metric, 
            self.train_cfg.early_stop_big_is_better, 
            self.train_cfg.early_stop_patience, 
            self.train_cfg.early_stop_min_delta
        )
        
        self.train_meter = MultiAverageMeter()

        # make dataloader an infinite step iterator
        self.train_batch_iter = batch_iter(self.train_dataloader)

        #step tracking
        self.total_step = 0
        self.opt_step = 0

    
    # step/interval tracking:
    
    def step(self) -> None:
        self.total_step += 1
        self.opt_step = self.total_step // self.train_cfg.grad_accum

    def hit_step_interval(self, interval: int) -> bool:
                
        if not (self.opt_step % interval == 0):
            return False

        if not (self.total_step % self.train_cfg.grad_accum == 0):
            return False
        
        return True
    
    # logging stuff:
    
    def log(self, metrics: dict, step: int, mode: Literal['train', 'eval', 'test']) -> None:
        
        wandb_metrics = wandb_format_metrics(metrics, mode)
        self.accelerator.log(wandb_metrics, step)

        info_metrics = {
            m: v for m, v in desection(metrics).items()
            if m in self.cfg.logging.info_metrics
        }
        self.logger.info('[ %-5s | Step: %-6d | %s ]', mode.capitalize(), self.opt_step, cmdline_format_metrics(info_metrics))
        self.logger.debug('[ %-5s | Step: %-6d | %s ]', mode.capitalize(), self.opt_step, cmdline_format_metrics(metrics))

    def summary(
            self, metrics: dict, mode: Literal['train', 'eval', 'test']
        ) -> None:

        self.logger.info('[ %-5s | Summary | %s]', mode.capitalize(), cmdline_format_metrics(metrics))

        import wandb
        if not self.accelerator.is_main_process:
            return
        
        wandb_run = self.accelerator.get_tracker('wandb', unwrap=True) # type: ignore 
        if wandb_run is None or not isinstance(wandb_run, wandb.Run):
            return

        wandb_metrics = wandb_format_metrics(metrics, mode)
        for metric, val in wandb_metrics.items():
            wandb_run.summary[metric] = val 

    def train_init_log(self) -> None:

        self.logger.info('Starting training')
        self.logger.info('Model: %s', self.model.__class__.__name__)
        self.logger.info('Parameters: total=%d', get_param_count(self.model))
        self.logger.info('Config:\n%s',OmegaConf.to_yaml(self.cfg, resolve=True))
        self.logger.info('Full model:\n%s', self.model)
    
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
                metrics |= section({'grad_norm': grad_norm.item()}, 'optim')

        self.optimiser.step()
        self.scheduler.step()
        self.optimiser.zero_grad()

        metrics |= section(out.get('metrics', {}), 'model')
        metrics |= section(loss_metrics, 'loss')
        metrics |= section(self.callbacks.on_train_step_end(self, out, batch), 'callbacks')
        metrics |= section({'lr': float(self.optimiser.param_groups[0]['lr'])}, 'optim')

        return metrics

    @torch.inference_mode()
    def evaluate(
        self,
        dataloader: DataLoader | Iterator[dict[str, Any]],
        forward_args: dict[str, Any],
        mode: Literal['eval', 'test']
        ) -> dict[str, float]:

        self.model.eval()
        eval_metrics = MultiAverageMeter()
        for batch in iter(dataloader):

            batch_metrics = {}

            out: dict = self.model(**batch, **forward_args)
            _, loss_metrics = self.loss_fn(out, batch)

            batch_metrics |= section(out.get('metrics', {}), 'model')
            batch_metrics |= section(loss_metrics, 'loss')
            if mode == 'eval':
                batch_metrics |= section(
                    self.callbacks.on_eval_step_end(self, out, batch), 'callbacks'
                )
            else:
                batch_metrics |= section(
                    self.callbacks.on_test_step_end(self, out, batch), 'callbacks'
                )

            eval_metrics.update(batch_metrics, n=get_batch_dict_size(batch))
        
        return eval_metrics.get_averages()

    # main entry point:
    
    def train(self) -> dict[str, float]:

        self.train_init_log()

        while self.opt_step < self.train_cfg.n_steps:
            
            self.step()
            batch = next(self.train_batch_iter)
            
            with self.accelerator.accumulate(self.model):
                train_step_metrics = self.train_step(
                    batch, self.forward_args, self.train_cfg.grad_clip
                )
            
            self.train_meter.update(train_step_metrics, n=get_batch_dict_size(batch))

            if self.hit_step_interval(self.cfg.logging.eval_log_interval):
                eval_metrics = self.evaluate(
                    self.eval_dataloader, self.forward_args, mode='eval'
                )
                self.log(eval_metrics, self.opt_step, mode='eval')
                early_stopping_summary = self.early_stopping.update(
                    desection(eval_metrics), self.opt_step
                )
                if early_stopping_summary['should_stop']:
                    self.logger.info(
                        'early stopping triggered | best step: %s | num_bad_steps: %s', 
                        early_stopping_summary['best_step'],
                        early_stopping_summary['num_bad_steps'],
                        )
                    break

            if self.hit_step_interval(self.cfg.logging.train_log_interval):
                train_metrics = self.train_meter.get_averages()
                self.log(train_metrics, self.opt_step, mode='train')
                
                self.train_meter.reset()
        
        self.early_stopping.load_best_model() 
        self.logger.info('loading best model')
        if self.cfg.logging.save_best:
            self.early_stopping.save_best_model(self.out_dir)
            
        es_best_state = self.early_stopping.get_best_stats()
        self.summary(es_best_state, 'eval')
        
        test_metrics = self.evaluate(
            self.test_dataloader, self.forward_args, mode='test'
            )
        self.summary(test_metrics, 'test')

        self.callbacks.on_train_end(self)

        return es_best_state | test_metrics