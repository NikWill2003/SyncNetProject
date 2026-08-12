import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from src import (
    register_configs,
    build_dataloaders,
    build_model,
    build_optim,
    build_lr_scheduler,
    build_loss_fn,
    build_callbacks,
)

from src.core import Config
from src.training import Trainer, get_wandb_init
from src.training.utils import set_seed

register_configs()

logger = get_logger(__name__)

@hydra.main(config_path='conf', config_name='config', version_base='1.3')
def main(cfg: Config) -> float:

    out_dir = HydraConfig.get().runtime.output_dir

    set_seed(cfg.train.seed)

    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        gradient_accumulation_steps=cfg.train.grad_accum,
        log_with='wandb' if cfg.wandb.enabled else None,
        project_config=ProjectConfiguration(project_dir=out_dir),
    )

    device = str(accelerator.device)
    dataloaders = build_dataloaders(cfg, device)
    model = build_model(cfg)

    if cfg.train.compile_model:
        model.compile()

    optimiser = build_optim(model, cfg.optim)
    scheduler = build_lr_scheduler(optimiser, cfg.train.n_steps, cfg.optim)
    loss_fn = build_loss_fn(cfg)
    callbacks = build_callbacks(cfg)


    if cfg.wandb.enabled:
        if cfg.wandb.project_name is None:
            raise ValueError('project name must be specified when wandb is enabled')
        
        accelerator.init_trackers(
            project_name=cfg.wandb.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
            init_kwargs={'wandb': get_wandb_init(cfg, out_dir)},
        )

    try:
        trainer = Trainer(
            cfg=cfg,
            out_dir=out_dir,
            logger=logger,
            model=model,
            dataloaders=dataloaders,
            optimiser=optimiser,
            scheduler=scheduler,
            accelerator=accelerator,
            callbacks=callbacks,
            loss_fn=loss_fn,
        )

        final_metrics = trainer.train()
    
    finally:
        accelerator.end_training()

    return float(final_metrics.get(f'best_{cfg.train.early_stop_metric}', 0.0))


if __name__ == '__main__':
    main()
