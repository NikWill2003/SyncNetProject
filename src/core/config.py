from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING



@dataclass
class DataConfig:
    name: str = MISSING
    root: str = MISSING
    dir: str = MISSING


@dataclass
class ModelConfig:
    # Models must run on the batch alone: `model(batch)`. Static
    # forward behaviour is a field on the model config; a callback that
    # needs a different forward (traces, t_override) calls the model
    # itself.
    name: str = MISSING


@dataclass
class CallbackConfig:
    name: str = MISSING


@dataclass
class TrainConfig:
    seed: int = 0

    n_steps: int = 1_000
    train_bs: int = 256
    val_bs: int = 1024

    early_stop_metric: str = 'loss'  # options: 'loss', 'accuracy'
    early_stop_big_is_better: bool = False
    # None -> never stop early (infinite patience). Best-model tracking
    # stays on regardless: main() returns `best_{early_stop_metric}` as the
    # hydra sweeper objective, so disabling the manager itself would make
    # every sweep silently optimise 0.0.
    early_stop_patience: int | None = 10
    early_stop_min_delta: float = 0.0

    mixed_precision: str = 'no'  # options: 'no', 'bf16', 'fp16'
    compile_model: bool = False
    grad_accum: int = 1
    grad_clip: float = 1.0

    loader_mode: str = 'gpu_cached'  # options: 'dataloader', 'gpu_cached'
    num_workers: int = 0


@dataclass
class LoggingConfig:
    eval_log_interval: int = 500
    train_log_interval: int = 100

    info_metrics: list[str] = field(
        default_factory=lambda: ['loss']
    )
    save_best: bool = False


@dataclass
class WandBConfig:
    enabled: bool = False
    project_name: str | None = None
    entity: str | None = None
    tags: list[str] = field(default_factory=list)
    run_name: str | None = None # prefer not to override and just allow for auto-naming

@dataclass
class OptimConfig:
    optimiser: str = 'adamw'
    lr: float = 3e-4
    weight_decay: float = 0.0

    lr_scheduler: str = 'cosine_annealing' # warmup_cosine or cosine_annealing or constant
    lr_scheduler_params: dict[str, Any] = field(default_factory=dict)




@dataclass
class Config:

    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    dataset: DataConfig = MISSING
    model: ModelConfig = MISSING
    callbacks: list[Any] = MISSING
