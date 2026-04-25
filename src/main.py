from __future__ import annotations

import os
import logging
import argparse
from typing import Dict

from paths import paths
from task import Task

import setup
import evaluate
import plots
import analyze
import xai
import sankey


TASK_REGISTRY: Dict[str, Task] = {
    "setup": setup.get_task(),
    "evaluate": evaluate.get_task(),
    "plot": plots.get_task(),
    "analyze": analyze.get_task(),
    "xai": xai.get_task(),
    "sankey": sankey.get_task(),
}


def setup_logging():
    os.makedirs(paths.logs, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-35s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(paths.log_file), logging.StreamHandler()],
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
        logger.info(f"Task '{args.task}' initialization")

        task = TASK_REGISTRY.get(args.task)
        if not task:
            logger.error(f"[ERROR] Task '{args.task}' not found in registry.")
            return 1

        task.run(args)
        logger.info(f"Task '{args.task}' completed")
        return 0
    except Exception as e:
        logger.error(f"[ERROR] Task failed with error: {e}")
        return 1


if __name__ == "__main__":
    main()
