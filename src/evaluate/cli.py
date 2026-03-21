from __future__ import annotations

import argparse
import logging

from const import DEFAULT_DATAPATH, DEFAULT_MODEL_NAME, DEFAULT_OUTPUT_PATH
from model import get_model
from task import Task
from experiment import read_experiments
from .evaluate import run_evaluation

__all__ = ["get_task", "register", "run"]

TASK_NAME = "evaluate"

logger = logging.getLogger(TASK_NAME)


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser("evaluate", help="Run model evaluation")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, type=str, help="Model name (e.g resnet152)")
    parser.add_argument("--data-path", default=DEFAULT_DATAPATH, type=str, help="Dataset path - data/")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, type=str)
    parser.add_argument("--experiments", default="experiments/experiments.yaml" ,type=str, help="Experiments")
    parser.add_argument("--run-single", type=str, help="Run specific experiment by name")
    parser.add_argument("--run-batch", type=str, help="Run specific batch of experiment by name")
    parser.add_argument("--sync-drive", action="store_true", help="Sync results to Google Drive")
    # fmt: on


def run(args: argparse.Namespace):
    logger.info(f"Starting evaluation of {args.model}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output path: {args.output_path}")

    model = get_model(args.model)
    experiments = read_experiments(args)
    logger.info(f"Found {len(experiments)} experiments")

    run_evaluation(
        config=args,
        model=model,
        experiments=experiments,
    )

    logger.info("Evaluation finished.")
