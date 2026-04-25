from __future__ import annotations

import importlib
from .base import BaseTask


def run_analysis(task: BaseTask, output_dir: str):
    try:
        module = importlib.import_module(f"analyze.tasks.{task.type}")
    except ImportError:
        raise ValueError(f"Unsupported analysis type: {task.type}")
    else:
        module.run(task, output_dir)
