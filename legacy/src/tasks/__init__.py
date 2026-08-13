from ..core.registry import TaskSpec

from . import sort_of_clevr
from . import coalitions
from . import sqoop

ALL_TASKS: tuple[TaskSpec, ...] = (
    sort_of_clevr.TASK,
    coalitions.TASK,
    sqoop.TASK,
)

TASKS: dict[str, TaskSpec] = {task.name: task for task in ALL_TASKS}

__all__ = ['ALL_TASKS', 'TASKS']
