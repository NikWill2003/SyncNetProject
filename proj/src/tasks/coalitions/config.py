from __future__ import annotations

from dataclasses import dataclass

from ...core.config import DataConfig


@dataclass
class CoalitionsDataConfig(DataConfig):
    name: str = 'coalitions'
    seed: int = 1
    root: str = './data'
    dir: str = 'coalitions'

    # task shape
    n_modules: int = 4 # supported catalogues: 2, 4, 6
    K: int = 4 # small alphabet: integer-sum readout stays learnable
    family: str = 'all' # 'clusterable'|'nonclique'|'frustrated'|'all'
    increments: tuple[int, ...] = (1, 2, 3) # local-rule step sizes ('rule' mode)
    stream_mode: str = 'iid' # 'iid' (routing forced) | 'rule' (deterministic)

    # sizes / lengths
    train_size: int = 20000
    test_size: int = 2000
    T_train: int = 32
    T_test: int = 32 # set > T_train for length generalisation

    # episode schedule
    episodes_min: int = 1
    episodes_max: int = 3
    window_min: int = 4
    window_max: int = 8
    post_steps: int = 2 # steps after disconnect tagged POST

    # command channel & readout
    command_mode: str = 'sparse' # 'sparse' | 'dense'
    readout_mode: str = 'instant' # 'instant' | 'latch'
    readout_lag_max: int = 3 # latch mode only

    # rho ladder dimensions cached into the .npz for calibration
    rho_dims: tuple[int, ...] = (2, 3, 4)
