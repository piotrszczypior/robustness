from __future__ import annotations

import argparse
import logging

from paths import paths
from model import get_model
from task import Task
from .experiment import read_experiments
from .evaluate import run_evaluation

__all__ = ["get_task", "register", "run"]

TASK_NAME = "evaluate"

logger = logging.getLogger(TASK_NAME)


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser("evaluate", help="Run model evaluation")
    parser.add_argument("--model", default="resnet152", type=str, help="Model name (e.g resnet152)")
    parser.add_argument("--data-path", default=str(paths.data), type=str, help="Dataset path")
    parser.add_argument("--output-path", default=str(paths.results), type=str)
    parser.add_argument("--experiments", default=str(paths.experiments_file) ,type=str, help="Experiments")
    parser.add_argument("--run-single", type=str, help="Run specific experiment by name")
    parser.add_argument("--run-batch", type=str, help="Run specific batch of experiment by name")
    parser.add_argument("--sync-drive", action="store_true", help="Sync results to Google Drive")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--batch-size", help="Batch size", default=128)
    parser.add_argument("--num-workers", help="Number of workers", default=12)
    parser.add_argument("--extract-features", action="store_true")
    parser.add_argument("--device", type=str, help="Specific device to use for evaluation (e.g., cuda:0, cuda:1, cpu)")
    # fmt: on


def run(args: argparse.Namespace):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting evaluation of {args.model}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output path: {args.output_path}")

    try:
        model, transforms = get_model(args.model)
        experiments = read_experiments(args)
        logger.info(f"Found {len(experiments)} experiments")

        run_evaluation(
            args=args,
            model=model,
            experiments=experiments,
            transforms=transforms,
        )
    except Exception as e:
        logger.error(f"[ERROR] Evaluation failed: {e}")
        raise RuntimeError(f"Evaluation task failed: {e}") from e
