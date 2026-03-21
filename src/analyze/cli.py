from __future__ import annotations

import argparse
import logging

from const import DEFAULT_DATAPATH, DEFAULT_MODEL_NAME, DEFAULT_OUTPUT_PATH
from task import Task

__all__ = ["get_task", "register", "run"]

TASK_NAME = "analyze"

logger = logging.getLogger(TASK_NAME)


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser("analyze", help="Analyze results")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, type=str)
    parser.add_argument("--data-path", default=DEFAULT_DATAPATH, type=str, help="Dataset path - data/")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, type=str)
    parser.add_argument("--sync-drive", action="store_true", help="Sync results to Google Drive")
    # fmt: on


def run(args: argparse.Namespace):
    logger.info("Starting analysis...")
    # TODO: Implement analysis logic
    logger.info("Analysis finished.")
