"""Launch the real Hydra CLI for each command, but do no work.

`--cfg job` is not enough: it validates composition, not sweep syntax.
This runs the actual multirun launcher and counts the jobs it produces.
"""
import hydra
from src.core.registry import register_configs, build_model

register_configs()
COUNT = {'n': 0}


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    COUNT['n'] += 1
    build_model(cfg)          # catches a cell that composes but cannot build
    return 0.0


if __name__ == '__main__':
    main()
    print(f'JOBS_LAUNCHED {COUNT["n"]}')
