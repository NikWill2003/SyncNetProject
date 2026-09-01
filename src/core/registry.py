from __future__ import annotations

from typing import Callable, Iterator

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

from .config import Config
from .callbacks import CallBackList

OnDeviceIter = Iterator[dict[str, torch.Tensor]]
Loaders = tuple[DataLoader | OnDeviceIter, ...]
LossFn = Callable[[dict, dict], tuple[Tensor, dict[str, float]]]


def register_configs() -> None:
    from ..datasets import DATASETS
    from ..models import MODELS

    cs = ConfigStore.instance()

    cs.store(name='config_schema', node=Config)

    for dataset in DATASETS.values():
        cs.store(group='dataset', name=f'{dataset.NAME}_base',
                 node=dataset.DATA_CONFIG)

    for name, (config, _) in MODELS.items():
        cs.store(group='model', name=f'{name}_base', node=config)


def get_dataset(cfg: Config):
    from ..datasets import DATASETS

    dataset = DATASETS.get(cfg.dataset.name)
    if dataset is None:
        raise ValueError(
            f'unknown dataset: {cfg.dataset.name!r} (available: {sorted(DATASETS)})'
        )
    return dataset


def build_model(cfg: Config) -> nn.Module:
    from ..models import MODELS

    name = str(cfg.model.name)
    entry = MODELS.get(name)
    if entry is None:
        raise ValueError(
            f'unknown model: {name!r} (available: {sorted(MODELS)})'
        )
    _, model_class = entry
    dataset = get_dataset(cfg)
    return model_class.from_config(
        cfg.model, dataset.NAME, int(dataset.ANSWER_DIM)
        )


def build_callbacks(cfg: Config, model=None):

    from .callbacks import SHARED_CALLBACKS

    dataset = get_dataset(cfg)
    dataset_callbacks = getattr(dataset, 'CALLBACKS', {})

    overlap = set(SHARED_CALLBACKS) & set(dataset_callbacks)
    if overlap:
        raise ValueError(f'duplicate callback(s) in {dataset.NAME}: {sorted(overlap)}')

    available = {**SHARED_CALLBACKS, **dataset_callbacks}
    offered = frozenset(getattr(model, 'supported_callbacks', ()))

    callbacks = []
    for cb_cfg in cfg.callbacks:
        spec = available.get(cb_cfg.name)
        if spec is None:
            raise ValueError(f'unknown callback {cb_cfg.name!r}; available: {sorted(available)}')

        missing = spec.requires - offered
        if missing:
            raise ValueError(
                f'callback {cb_cfg.name!r} requires {sorted(missing)}, '
                f'but {type(model).__name__} supports {sorted(offered)}'
            )

        typed = OmegaConf.merge(OmegaConf.structured(spec.config), cb_cfg)
        callbacks.append(spec.callback_class.from_config(cfg, typed))  # type: ignore

    return CallBackList(callbacks)


def build_dataloaders(cfg: Config, device: str) -> Loaders:

    name = str(cfg.dataset.name)
    if name == 'sort_of_clevr':
        from ..datasets.soc.loader import build_dataloaders as _build
    elif name == 'sqoop':
        from ..datasets.sqoop.loader import build_dataloaders as _build
    else:
        raise ValueError(f'unknown dataset: {name!r}')
    return _build(cfg, device) # type: ignore


def build_loss_fn(cfg: Config) -> LossFn:

    ce_loss = nn.CrossEntropyLoss()
    
    def cross_entropy(
        out: dict, batch: dict,
        ) -> tuple[Tensor, dict[str, float]]:

        loss = ce_loss(out['logits'].float(), batch['answers'])
        return loss, {'loss': loss.item()}

    return cross_entropy


def prepare_dataset(cfg: Config) -> None:
    """Generate the on-disk splits. Dispatches like build_dataloaders: the
    generator belongs to the dataset, not to its spec."""
    name = str(cfg.dataset.name)
    if name == 'sort_of_clevr':
        from ..datasets.soc.generator import prepare_sort_of_clevr as _prepare
    elif name == 'sqoop':
        from ..datasets.sqoop.generator import prepare_sqoop as _prepare
    else:
        raise ValueError(f'unknown dataset: {name!r}')
    _prepare(cfg.dataset)
