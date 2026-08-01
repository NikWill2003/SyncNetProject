"""Build the on-disk dataset for a task.

Usage:
    python prepare_dataset.py task=sort_of_clevr dataset=sort_of_clevr/smoke
    python prepare_dataset.py task=sort_of_clevr
    python prepare_dataset.py task=coalitions
"""
import hydra

from src import register_configs
from src.core import Config
from src.core.registry import prepare_dataset

register_configs()


@hydra.main(config_path='conf', config_name='prepare', version_base='1.3')
def main(cfg: Config) -> None:
    prepare_dataset(cfg)


if __name__ == '__main__':
    main()
