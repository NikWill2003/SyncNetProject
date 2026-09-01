"""Dataset registry.

Each package holds `spec.py` (constants and DataConfig -- pure data),
`generator.py`, `translate.py`, `loader.py`, and `callbacks.py` for the
callbacks that need its question layout. `DATASETS` exposes the spec with
that dataset's callbacks attached.
"""

from .soc import callbacks as _soc_callbacks
from .soc import spec as soc
from .sqoop import callbacks as _sqoop_callbacks
from .sqoop import spec as sqoop

soc.CALLBACKS = _soc_callbacks.CALLBACKS
sqoop.CALLBACKS = _sqoop_callbacks.CALLBACKS

DATASETS = {soc.NAME: soc, sqoop.NAME: sqoop}
