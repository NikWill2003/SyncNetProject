from pathlib import Path

import hydra
from omegaconf import OmegaConf

from src import register_configs
from src.core.config import Config


register_configs()

@hydra.main(config_path='conf', config_name='config', version_base='1.3')
def main(cfg: Config) -> None:
    print('Config:\n%s',OmegaConf.to_yaml(cfg, resolve=True))

if __name__ == '__main__':
    main()
