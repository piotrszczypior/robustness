from __future__ import annotations

import argparse
import logging
from .settings import get_accuracy_scatter_plot_specs
from .plot import AccuracyToAccuracy, AccuracyToAccuracyDrop

__all__ = ["register", "run"]

PLOT_NAME = "accuracy_scatter"

logger = logging.getLogger(PLOT_NAME)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser(PLOT_NAME, help="Generate accuracy vs accuracy scatter plots")
    parser.add_argument("x_file", type=str, help="CSV file for the X axis")
    parser.add_argument("y_file", type=str, help="CSV file for the Y axis")
    parser.add_argument("--mode", default="default", choices=["default", "drop"], help="Plotting mode: 'default' for direct comparison, 'drop' for accuracy drop.")
    parser.add_argument("--data", default="results/")
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(accuracy_scatter_run=run)
    # fmt: on


def run(args: argparse.Namespace):
    logger.info(f"Generating accuracy scatter plot for {args.x_file} vs {args.y_file}")

    plot_spec = get_accuracy_scatter_plot_specs(args.x_file, args.y_file, args.mode)

    if args.mode == "drop":
        AccuracyToAccuracyDrop(config=plot_spec, data_dir=args.data).run()
    else:
        AccuracyToAccuracy(config=plot_spec, data_dir=args.data).run()
