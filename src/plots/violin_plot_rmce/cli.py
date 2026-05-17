from __future__ import annotations

import argparse
import logging
from .settings import get_violin_plot_rmce_specs
from .plot import ViolinRmCEPlot

__all__ = ["register", "run"]

PLOT_NAME = "violin_rmce"

logger = logging.getLogger(PLOT_NAME)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser(PLOT_NAME, help="Generate Violin plots for per-class RmCE distributions")
    parser.add_argument("--data", default="results/")
    parser.add_argument("--models", nargs="+", type=str, help="Specific models to plot")
    parser.add_argument("--corruptions", nargs="+", type=str, help="Specific corruptions to plot")
    parser.add_argument("--severities", nargs="+", type=int, help="Specific severities to plot")
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(violin_rmce_run=run)
    # fmt: on


def run(args: argparse.Namespace):
    logger.info("Generating violin RmCE plots")

    plots = get_violin_plot_rmce_specs(
        models=args.models, corruptions=args.corruptions, severities=args.severities
    )
    for cng in plots:
        ViolinRmCEPlot(config=cng, data_dir=args.data).run()
