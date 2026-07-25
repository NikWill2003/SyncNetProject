from typing import Any, Literal, Optional, TYPE_CHECKING
from pathlib import Path
import math
from collections import defaultdict

import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

if TYPE_CHECKING:
    from ..core.config import Config


def get_wandb_init(cfg: Config, out_dir: str) -> dict[str, Any]:
    """
    Naming:
        single run: {model}::{date}::{time}
        multirun: {model}::multirun::{sweep_stamp}::{job_num}

    Group:
        - only populates on multirun
        - concatenates the timestamp with the overrides excluding the seeds so that
          seeded runs can be grouped 

    Tags: 
        append: 
            - sweep:{stamp} (multirun if multirun)
            - model:{choice}
            - dataset:{name}
            - exp:{choice}
    """
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

        non_seed = [o for o in overrides if not o.startswith('train.seed=')]
        group = f'{time_stamp}|{",".join(sorted(non_seed))}'

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
        'group': cfg.wandb.group or group,
        'tags': list(cfg.wandb.tags) + auto_tags,
        'notes': ' '.join(overrides), 
        'job_type': 'train',
        'dir': out_dir,
    }

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
