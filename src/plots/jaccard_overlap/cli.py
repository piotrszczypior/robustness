from __future__ import annotations

import argparse
import logging
from .settings import get_jaccard_plot_specs
from .plot import DomainJaccardOverlapPlot

__all__ = ["register", "run"]

PLOT_NAME = "jaccard"

logger = logging.getLogger(PLOT_NAME)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser(PLOT_NAME, help="Generate Jaccard overlap plots for worst classes")
    parser.add_argument("--top-k", default=50, type=int, help="Number of worst classes to consider")
    parser.add_argument("--data", default="results/")
    parser.add_argument("--tail", default="worst", type=str)
    parser.add_argument("--corruptions", nargs="+", type=str, help="Specific corruptions to plot (overrides static list)")
    parser.add_argument("--severities", nargs="+", type=int, help="Specific severities to plot (overrides static list)")
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(jaccard_run=run)
    # fmt: on


def run(args: argparse.Namespace):
    logger.info(f"Generating Jaccard overlap plots for top {args.top_k}")

    plots = get_jaccard_plot_specs(
        top_k=args.top_k,
        corruptions=args.corruptions,
        severities=args.severities,
        tail=args.tail,
    )
    for cng in plots:
        DomainJaccardOverlapPlot(config=cng, data_dir=args.data).run()
