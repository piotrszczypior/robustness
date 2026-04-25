from __future__ import annotations

import argparse
import logging
from pathlib import Path

from task import Task
from .setup import setup_dataset
from paths import paths

__all__ = ["get_task", "register", "run"]

TASK_NAME = "setup"

logger = logging.getLogger(TASK_NAME)


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser("setup", help="Setup environment")
    parser.add_argument("--dataset", help="Datasets to prepare", required=True)
    parser.add_argument("--data-path", default=paths.data, type=str, help="Dataset path - data/")
    parser.add_argument("--archives", nargs="*", help="Specific archives to download (e.g. blur.tar)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    # fmt: on


def run(args: argparse.Namespace):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    data_path = Path(args.data_path)
    logger.info(f"Preparing data in {args.data_path}")

    try:
        setup_dataset(data_path, args.dataset, args.archives)
    except Exception as e:
        logger.error(f"[ERROR] Setup failed: {e}")
        raise RuntimeError(f"Setup task failed: {e}") from e
