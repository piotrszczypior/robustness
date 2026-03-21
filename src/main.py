from __future__ import annotations

import os
import logging
import argparse
from typing import Dict

from config import Config
from task import Task

import setup
import evaluate
import plots


TASK_REGISTRY: Dict[str, Task] = {
    "setup": setup.get_task(),
    "evaluate": evaluate.get_task(),
    "plot": plots.get_task(),
}


def setup_logging():
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(Config.LOGS_DIR, Config.LOG_FILE)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Robustness")
    subparsers = parser.add_subparsers(dest="task", required=True)

    for task in TASK_REGISTRY.values():
        task.register(subparsers)

    return parser.parse_args()


def main() -> int:
    logger = setup_logging()

    try:
        args = get_args()
        logger.info(f"Starting task: {args.task}")

        task = TASK_REGISTRY.get(args.task)
        if not task:
            logger.error(f"Task '{args.task}' not found in registry.")
            return 1

        task.run(args)
        logger.info(f"Task {args.task} completed successfully")
        return 0
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    main()
