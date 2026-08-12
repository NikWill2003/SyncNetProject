from __future__ import annotations

from typing import Any, Literal, Optional, TYPE_CHECKING, cast
from pathlib import Path
import math
from collections import defaultdict
import hashlib

import torch.nn as nn
from accelerate import Accelerator
import wandb
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

from .utils import get_param_count

if TYPE_CHECKING:
    from ..core.config import Config


def get_wandb_init(cfg: Config, out_dir: str) -> dict[str, Any]:
    
    hc = HydraConfig.get()

    choices = hc.runtime.choices
    model_choice = choices.get('model') or cfg.model.name
    exp_choice = choices.get('experiment')

    run_dir = Path(out_dir)
    overrides = list(hc.overrides.task)  

    if hc.mode == RunMode.MULTIRUN:
        # outputs/<dataset>/multirun/<stamp>/<job_num>
        time_stamp, job_num = run_dir.parent.name, run_dir.name
        name = f'{model_choice}::multirun::{time_stamp}::{job_num}'

        non_seed = sorted(o for o in overrides if not 'train.seed=' in o)
        non_seed_str = ",".join(non_seed)
        non_seed_hash = hashlib.sha256(non_seed_str.encode('utf-8')).hexdigest()[:12]

        group = f'{time_stamp}|seeded_rerun:{non_seed_hash}'

        auto_tags = [f'sweep:{time_stamp}']
    else:
        # outputs/<dataset>/<date>/<time>
        date, time = run_dir.parent.name, run_dir.name
        name = f'{model_choice}::{date}::{time}'

        group = None
        auto_tags = []

    auto_tags += [f'model:{model_choice}', f'dataset:{cfg.dataset.name}']
    if exp_choice:
        auto_tags.append(f'exp:{exp_choice}')

    return {
        'entity': cfg.wandb.entity,
        'name': cfg.wandb.run_name or name,
        'group': group,
        'tags': list(cfg.wandb.tags) + auto_tags,
        'notes': ' '.join(overrides), 
        'job_type': 'train',
        'dir': out_dir,
    }

def accelerate_init_wandb(
    cfg: Config,
    accelerator: Accelerator, 
    out_dir: str, 
    model: nn.Module
    ) -> None:
    """
    Run Name if not specified:
        single run: {model}::{date}::{time}
        multirun: {model}::multirun::{sweep_stamp}::{job_num}

    Group (reserved for seeded reruns):
        format: {timestamp}|seeded_rerun:{hashed_overides}
        Only populated when a run is a multirun; appends hashed multirun overrides
        to the multirun timestamps so that seeded reruns can be grouped

    Append Tags: 
        - sweep:{stamp} (only if multirun)
        - model:{choice}
        - dataset:{name}
        - exp:{choice}
    """

    if cfg.wandb.project_name is None:
        raise ValueError('project name must be specified when wandb is enabled')
    
    dict_cfg = dict[str, Any], OmegaConf.to_container(cfg, resolve=True) 
    dict_cfg['n_params'] = get_param_count(model) # type: ignore

    choices = HydraConfig.get().runtime.choices
    if experiemnt := choices.get('experiment'): dict_cfg['experiment'] = experiemnt # type: ignore

    accelerator.init_trackers(
        project_name=cfg.wandb.project_name,
        config=dict_cfg,  # type: ignore
        init_kwargs={'wandb': get_wandb_init(cfg, out_dir)},
    )

def section(metrics: dict[str, float] | None, name: str) -> dict[str, float]:
    # prefix metrics to section them in wandb
    return {f'{name}/{k}': v for k, v in (metrics or {}).items()}


def desection(metrics: dict[str, float]) -> dict[str, float]:
    # strip the prefixes used for organising metrics in wandb 
    out: dict[str, float] = {}
    origin: dict[str, str] = {}

    for key, val in metrics.items():
        name = key.rsplit('/', 1)[-1]
        if name in out:
            raise KeyError(
                f'metric name collision: {origin[name]!r} and {key!r}, rename one of them'
            )
        out[name] = val
        origin[name] = key

    return out

def wandb_format_metrics(
        metrics: dict[str, Any],
        mode: Literal['train', 'eval', 'test']
        ) -> dict[str, Any]:
    # append the mode onto the key, i.e. loss/cross_entropy -> train_loss/cross_entropy
    out = {}
    for key, val in metrics.items():
        if '/' in key:
            sec, name = key.split('/', 1)
            out[f'{mode}_{sec}/{name}'] = val
        else:
            out[f'{mode}/{key}'] = val
    return out


def cmdline_format_metrics(metrics: dict) -> str:

    # format: "| metric : val | metric : val | metric : val "
    return '|'.join(
        [f' {m} : {v:.4f} ' for m, v in metrics.items() if isinstance(v, (float, int))]
        )


def wandb_finish(wandb_run: Optional[wandb.Run]) -> None:

    if wandb_run is not None:
        wandb_run.finish()


class MultiAverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self._total = defaultdict(float)
        self._count = defaultdict(float)

    def update(self, metric_dict: dict[str, float], n: int = 1) -> None:
        if n <= 0:
            return

        for metric, val in metric_dict.items():
            if not math.isfinite(val):
                continue

            self._total[metric] += val * n
            self._count[metric] += n

    def get_averages(self) -> dict[str, float]:
        return {
            metric: self._total[metric] / self._count[metric]
            for metric in self._count
            }
