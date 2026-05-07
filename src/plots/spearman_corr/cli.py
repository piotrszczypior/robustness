from __future__ import annotations

import argparse
import logging
from .settings import get_spearman_plot_specs
from .plot import DomainSpearmanRankPlot

__all__ = ["register", "run"]

PLOT_NAME = "spearman"

logger = logging.getLogger(PLOT_NAME)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser(PLOT_NAME, help="Generate Spearman correlation plots for domain accuracy drops")
    parser.add_argument("--metric", default="drop", choices=["drop", "rank"], help="Metric to use for Spearman correlation")
    parser.add_argument("--data", default="results/")
    parser.add_argument("--corruptions", nargs="+", type=str, help="Specific corruptions to plot (overrides static list)")
    parser.add_argument("--severities", nargs="+", type=int, help="Specific severities to plot (overrides static list)")
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(spearman_run=run)
    # fmt: on


def run(args: argparse.Namespace):
    logger.info(f"Generating spearman plots for metric: {args.metric}")

    plots = get_spearman_plot_specs(
        metric_type=args.metric,
        corruptions=args.corruptions,
        severities=args.severities
    )
    for cng in plots:
        DomainSpearmanRankPlot(config=cng, data_dir=args.data).run()
