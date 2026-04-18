from __future__ import annotations

import argparse
import logging

from task import Task
from .analyses import (
    generate_accuracy_drop_tasks,
    generate_common_fragile_tasks,
    generate_fragile_class_tasks,
)
from .base import BaseTask
from .core import run_analysis

logger = logging.getLogger(__name__)

TASK_NAME = "analyze"

ANALYSES: list[BaseTask] = [
    *generate_fragile_class_tasks(),
    *generate_common_fragile_tasks(),
    *generate_accuracy_drop_tasks(),
]


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser("analyze", help="Analyze results")
    parser.add_argument("--type", type=str, help="Filter by task type")
    parser.add_argument("--data", default="results/", type=str, help="Data directory with csv files")
    parser.add_argument("--output", default="analysis/results/", type=str, help="Data directory with csv files")
    parser.add_argument("--sync-drive", action="store_true", help="Sync results to Google Drive")
    parser.add_argument("--debug", action="store_true", help="Skip plot generation")
    # fmt: on


def run(args: argparse.Namespace):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Loading tasks")
    tasks = [t for t in ANALYSES if not args.type or t.type == args.type]
    total = len(tasks)

    if total == 0:
        logger.warning("No valid configurations found.")
        return

    logger.info(f"Found {total} tasks")
    return
    for i, spec in enumerate(tasks, 1):
        logger.info(f"[{i}/{total}] Analysing '{spec.name}'...")
        run_analysis(spec, args.output)
