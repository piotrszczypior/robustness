from __future__ import annotations

import argparse
import logging

from config import Config
from task import Task
from .gradcam import run_gradcam

__all__ = ["get_task", "register", "run"]

TASK_NAME = "xai"

logger = logging.getLogger(TASK_NAME)


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(TASK_NAME, help="Run XAI analysis (GradCAM)")
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g. resnet152)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Dataset name (e.g. blur_defocus_blur_1)",
    )
    parser.add_argument(
        "--data-path", default=Config.DATA_ROOT, type=str, help="Base data path"
    )
    parser.add_argument(
        "--synset", required=True, type=str, help="ImageNet synset ID (e.g. n01440764)"
    )
    parser.add_argument(
        "--output-dir",
        default="images/gradcam",
        type=str,
        help="Output directory for heatmaps",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")


def run(args: argparse.Namespace):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting XAI analysis for model: {args.model}")
    logger.info(f"Dataset alias: {args.dataset}")
    logger.info(f"Synset: {args.synset}")

    try:
        run_gradcam(
            model_name=args.model,
            dataset_alias=args.dataset,
            synset=args.synset,
            output_dir=args.output_dir,
        )

    except Exception as e:
        logger.error(f"[ERROR] XAI analysis failed: {e}")
        raise RuntimeError(f"XAI task failed: {e}") from e
