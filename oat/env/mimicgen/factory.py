"""Task IDs and immutable source HDF5 environment arguments."""
import copy
import json
from pathlib import Path

TASKS = ('stack_d0', 'square_d0', 'coffee_d0', 'mug_cleanup_d0')
METADATA = json.loads(Path(__file__).with_name('task_metadata.json').read_text())


def get_subtasks(task_name='mt4'):
    if task_name in ('mt4', 'mimicgen-mt4'):
        return list(TASKS)
    if task_name not in TASKS:
        raise ValueError(f'Unknown MimicGen task: {task_name}')
    return [task_name]


def get_task(task_name):
    if task_name not in TASKS:
        raise ValueError(f'Unknown MimicGen task: {task_name}')
    return copy.deepcopy(METADATA[task_name])
