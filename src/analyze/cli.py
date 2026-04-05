from __future__ import annotations

import argparse
import logging

from task import Task

__all__ = ["get_task", "register", "run"]

TASK_NAME = "plot"

logger = logging.getLogger(TASK_NAME)


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser("analyze", help="")
    parser.add_argument("--debug", action="store_true", help="Skip plot generation")
    # fmt: on


def run(args: argparse.Namespace):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting Analyze Task")
