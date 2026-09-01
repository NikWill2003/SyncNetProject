"""Build the on-disk splits for a dataset, without training.

The loaders generate on demand too, so this is only needed to warm a
cache ahead of a sweep.

    python -m scripts.prepare_dataset task=sort_of_clevr
    python -m scripts.prepare_dataset task=sqoop dataset.rhs_variety=4
"""
import hydra

from src.core import Config, register_configs
from src.core.registry import prepare_dataset

register_configs()


@hydra.main(config_path='../conf', config_name='prepare', version_base='1.3')
def main(cfg: Config) -> None:
    prepare_dataset(cfg)


if __name__ == '__main__':
    main()
