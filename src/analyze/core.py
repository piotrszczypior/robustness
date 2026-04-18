from __future__ import annotations

import importlib
from .base import BaseTask


def run_analysis(task: BaseTask, output_dir: str):
    try:
        module = importlib.import_module(f"analyze.tasks.{task.type}")
        module.run(task, output_dir)
    except ImportError:
        raise ValueError(f"Unsupported analysis type: {task.type}")
