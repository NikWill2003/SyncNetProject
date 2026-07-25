import hydra

from src.training.utils import set_seed
from src.core import Config
from src import (
    register_configs,
    prepare_dataset,
    )

register_configs()

@hydra.main(config_path='conf', config_name='prepare', version_base='1.3')
def main(cfg: Config):

    print(f'preparing dataset: {cfg.dataset.name}')
    print('dataset configuration:')
    print(f'{cfg.dataset}')

    prepare_dataset(cfg)

if __name__ == '__main__':
    main()