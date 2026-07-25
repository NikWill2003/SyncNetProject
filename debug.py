from pathlib import Path

import hydra
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger

from src import (
    register_configs,
    build_dataloaders,
    build_model,
    build_optim,
    build_lr_scheduler,
    build_loss_fn,
)

from src.core import Config
from src.training import DebugTrainer
from src.training.utils import set_seed

register_configs()

logger = get_logger(__name__)

@hydra.main(config_path='conf', config_name='debug', version_base='1.3')
def main(cfg: Config) -> None:
    
    set_seed(cfg.train.seed)

    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        gradient_accumulation_steps=cfg.train.grad_accum,
    )

    device = str(accelerator.device)

    dataloaders = build_dataloaders(cfg, device)
    model = build_model(cfg)

    if cfg.train.compile_model:
        model.compile()

    optimiser = build_optim(model, cfg.optim)
    scheduler = build_lr_scheduler(optimiser, cfg.train.n_steps, cfg.optim)
    loss_fn = build_loss_fn(cfg)

    forward_args = OmegaConf.to_container(cfg.model.forward_args, resolve=True)
    
    try:
        trainer = DebugTrainer(
            cfg=cfg,
            logger=logger,
            model=model,
            dataloaders=dataloaders,
            forward_args=forward_args,  # type: ignore
            optimiser=optimiser,
            scheduler=scheduler,
            accelerator=accelerator,
            loss_fn=loss_fn,
        )

        trainer.debug_model()
    
    finally:
        accelerator.end_training()

if __name__ == '__main__':
    main()
