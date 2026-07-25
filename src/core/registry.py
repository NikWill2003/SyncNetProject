"""Task registry: specs describing each task and the builders that turn a
resolved config into runtime objects (model, dataloaders, callbacks, loss).

A task contributes a `TaskSpec` (see `tasks/<task>/__init__.py`); everything
else in the codebase goes through the builders below and never imports task
packages directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

from .config import Config, DataConfig, ModelConfig, CallbackConfig
from .callbacks import BaseCallBack, CallBackList
from .encoders import PatchifyEncoderConfig, CNNEncoderConfig

OnDeviceIter = Iterator[dict[str, torch.Tensor]]
Loaders = tuple[DataLoader | OnDeviceIter, ...]

# A loss fn maps (model output dict, batch dict) to (loss, metrics).
# Builders are zero-arg so construction stays lazy.
LossFn = Callable[[dict, dict], tuple[Tensor, dict[str, float]]]
LossBuilder = Callable[[], LossFn]


@dataclass(frozen=True)
class ModelSpec:
    config: type[ModelConfig]
    # must expose .from_config(model_cfg, data_cfg) -> model
    model_class: type[nn.Module]


@dataclass(frozen=True)
class CallbackSpec:
    config: type[CallbackConfig]
    # must expose .from_config(cfg, cb_cfg) -> callback
    callback_class: type[BaseCallBack]


@dataclass(frozen=True)
class TaskSpec:
    name: str  # matches cfg.dataset.name, e.g. 'sort_of_clevr'

    data_config: type[DataConfig]
    # (cfg, device) -> (train, val, test) loaders
    dataloader_builder: Callable[[Config, str], Loaders]
    # build the on-disk dataset from its data config
    prepare: Callable[[Any], None]

    models: dict[str, ModelSpec]
    callbacks: dict[str, CallbackSpec]

    loss_builder: LossBuilder


def _tasks() -> dict[str, TaskSpec]:
    # lazy: tasks import from core, so core must not import tasks at
    # module level
    from ..tasks import TASKS
    return TASKS


def get_task(cfg: Config) -> TaskSpec:
    task = _tasks().get(cfg.dataset.name)
    if task is None:
        raise ValueError(
            f'unknown dataset: {cfg.dataset.name!r} '
            f'(registered: {sorted(_tasks())})'
        )
    return task


def register_configs() -> None:
    cs = ConfigStore.instance()

    cs.store(name='config_schema', node=Config)

    for task in _tasks().values():

        cs.store(
            group='dataset',
            name=f'{task.name}_base',
            node=task.data_config,
        )

        for model_name, spec in task.models.items():
            cs.store(
                group='model',
                name=f'{model_name}_base',
                node=spec.config,
            )

    cs.store(
        group='model/encoder',
        name='patchify_encoder_base',
        node=PatchifyEncoderConfig,
        package='model.encoder_cfg',
    )

    cs.store(
        group='model/encoder',
        name='cnn_encoder_base',
        node=CNNEncoderConfig,
        package='model.encoder_cfg',
    )


def build_model(cfg: Config) -> nn.Module:
    task = get_task(cfg)

    spec = task.models.get(cfg.model.name)
    if spec is None:
        raise ValueError(
            f'{cfg.model.name!r} is not a supported model for {task.name} '
            f'(supported: {sorted(task.models)})'
        )

    return spec.model_class.from_config(cfg.model, cfg.dataset)  # type: ignore[attr-defined]


def build_callbacks(cfg: Config) -> CallBackList:
    task = get_task(cfg)

    callbacks = []
    for cb_cfg in cfg.callbacks:

        spec = task.callbacks.get(cb_cfg.name)
        if spec is None:
            raise ValueError(
                f'unrecognised callback {cb_cfg.name!r} '
                f'for dataset {cfg.dataset.name!r} '
                f'(supported: {sorted(task.callbacks)})'
            )

        # validate + type the raw yaml entry against the callback's config
        # schema (unknown keys or wrong types raise here, at build time)
        typed_cb_cfg = OmegaConf.merge(OmegaConf.structured(spec.config), cb_cfg)

        callbacks.append(spec.callback_class.from_config(cfg, typed_cb_cfg))

    return CallBackList(callbacks)


def build_dataloaders(cfg: Config, device: str) -> Loaders:
    return get_task(cfg).dataloader_builder(cfg, device)


def build_loss_fn(cfg: Config) -> LossFn:
    loss_fn = get_task(cfg).loss_builder()

    def with_total(out: dict, batch: dict) -> tuple[Tensor, dict[str, float]]:
        loss, metrics = loss_fn(out, batch)
        # guarantee the 'loss' key the trainer / early stopping rely on,
        # so task losses only report their own named terms
        metrics = dict(metrics)
        metrics['loss'] = loss.item()
        return loss, metrics

    return with_total


def prepare_dataset(cfg: Config) -> None:
    get_task(cfg).prepare(cfg.dataset)
