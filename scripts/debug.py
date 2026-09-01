"""Train on one batch for a fixed number of steps. No eval, no test.

Same construction path as main.py -- it imports build_components from
there, so only the trainer differs.

    python -m scripts.debug
    python -m scripts.debug task=sqoop model=sqoop/relnet
"""

import hydra
from accelerate.logging import get_logger

from src.core import Config, register_configs
from src.training import DebugTrainer

from .main import build_components

register_configs()

logger = get_logger(__name__)


@hydra.main(config_path='../conf', config_name='debug', version_base='1.3')
def debug(cfg: Config) -> float:

    components = build_components(cfg)

    try:
        trainer = DebugTrainer(cfg=cfg, logger=logger, **components)
        final_metrics = trainer.train()

    finally:
        components['accelerator'].end_training()

    return float(final_metrics.get('loss', 0.0))


if __name__ == '__main__':
    debug()
