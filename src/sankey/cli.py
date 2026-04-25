from __future__ import annotations

import argparse
import logging
from pathlib import Path

from paths import paths
from task import Task
from .plot import generate_sankey_plots

__all__ = ["get_task", "register", "run"]

TASK_NAME = "sankey"

logger = logging.getLogger(TASK_NAME)


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser("sankey", help="Generate Sankey plots from analysis data")
    parser.add_argument("--input-file", required=True, type=str, help="Input JSON file with analysis data")
    parser.add_argument("--output-dir", default=str(paths.images), type=str, help="Directory to save plots")
    parser.add_argument("--min-count", default=1, type=int, help="Minimum count for a data point to be included in the plot")
    parser.add_argument("--title", type=str, help="Plot title")
    # fmt: on


def run(args: argparse.Namespace):
    logger.info("Starting Sankey Plot Generation Task")

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir) / "sankey"

    if not input_file.is_file():
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        generate_sankey_plots(
            input_file=input_file,
            output_dir=output_dir,
            min_count=args.min_count,
            title=args.title,
        )
    except Exception as e:
        logger.error(f"[ERROR] Sankey plot generation failed: {e}")
        raise RuntimeError(f"Sankey plot generation task failed: {e}") from e
