from __future__ import annotations

import argparse
import logging

from task import Task
from .xai import run_xai
from paths import paths

__all__ = ["get_task", "register", "run"]

TASK_NAME = "xai"

logger = logging.getLogger(TASK_NAME)


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(TASK_NAME, help="Run XAI analysis (GradCAM)")
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Model name (e.g. resnet152)",
    )
    parser.add_argument(
        "--datasets",
        required=True,
        nargs="+",
        type=str,
        help="Dataset name(s) (e.g. imagenet imagenet_c_defocus_blur_1)",
    )
    parser.add_argument(
        "--data-path", default=paths.data, type=str, help="Base data path"
    )
    parser.add_argument(
        "--synset", required=True, type=str, help="ImageNet synset ID (e.g. n01440764)"
    )
    parser.add_argument(
        "--output-dir",
        default="xai/",
        type=str,
        help="Output directory for heatmaps",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")


def run(args: argparse.Namespace):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting XAI analysis for model: {args.model}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Synset: {args.synset}")

    try:
        run_xai(
            model_name=args.model,
            dataset_aliases=args.datasets,
            synset=args.synset,
            output_dir=args.output_dir,
        )

    except Exception as e:
        logger.error(f"[ERROR] XAI analysis failed: {e}")
        raise RuntimeError(f"XAI task failed: {e}") from e
