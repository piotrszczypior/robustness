from __future__ import annotations

import argparse
import logging
from .settings import get_class_degradation_plot_specs
from .plot import SortedIndexClassDegradation

__all__ = ["register", "run"]

PLOT_NAME = "class_degradation"

logger = logging.getLogger(PLOT_NAME)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser(PLOT_NAME, help="Generate class degradation plots")
    parser.add_argument("--baseline-label", type=str, required=True)
    parser.add_argument("--baseline-data", type=str, required=True)
    parser.add_argument("--degraded-label", type=str, required=True)
    parser.add_argument("--degraded-data", type=str, required=True)
    parser.add_argument("--data", default="results/")
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(class_degradation_run=run)
    # fmt: on


def run(args: argparse.Namespace):
    logger.info(
        f"Generating class degradation plot for {args.baseline_data} vs {args.degraded_data}"
    )

    plot_spec = get_class_degradation_plot_specs(
        baseline_label=args.baseline_label,
        baseline_data=args.baseline_data,
        degraded_label=args.degraded_label,
        degraded_data=args.degraded_data,
    )

    SortedIndexClassDegradation(config=plot_spec, data_dir=args.data).run()
