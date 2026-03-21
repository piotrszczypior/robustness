from __future__ import annotations

import argparse
import logging
from pathlib import Path

from task import Task
from .setup import setup_dataset
from const import DEFAULT_DATAPATH

__all__ = ["get_task", "register", "run"]

TASK_NAME = "setup"

logger = logging.getLogger(TASK_NAME)


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser("setup", help="Setup environment")
    parser.add_argument("--dataset", help="Datasets to prepare", required=True)
    parser.add_argument("--data-path", default=DEFAULT_DATAPATH, type=str, help="Dataset path - data/")
    parser.add_argument("--archives", nargs="*", help="Specific archives to download (e.g. blur.tar)")
    # fmt: on


def run(args: argparse.Namespace):
    data_path = Path(args.data_path)
    logger.info(f"Preparing data in {args.data_path}")

    setup_dataset(data_path, args.dataset, args.archives)
    logger.info("Data preparation finished.")
