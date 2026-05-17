from __future__ import annotations

import argparse
import logging
from .settings import get_violin_plot_specs
from .plot import ViolinPlot

__all__ = ["register", "run"]

PLOT_NAME = "violin"

logger = logging.getLogger(PLOT_NAME)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser(PLOT_NAME, help="Generate Violin plots for per-class accuracy distributions")
    parser.add_argument("--mode", default="single", choices=["single", "collage"], help="Plotting mode: single (one plot per model) or collage (all models in one plot)")
    parser.add_argument("--data", default="results/")
    parser.add_argument("--models", nargs="+", type=str, help="Specific models to plot")
    parser.add_argument("--corruptions", nargs="+", type=str, help="Specific corruptions to plot")
    parser.add_argument("--severities", nargs="+", type=int, help="Specific severities to plot")
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(violin_run=run)
    # fmt: on


def run(args: argparse.Namespace):
    logger.info(f"Generating violin plots in mode: {args.mode}")

    plots = get_violin_plot_specs(
        mode=args.mode,
        models=args.models,
        corruptions=args.corruptions,
        severities=args.severities,
    )
    for cng in plots:
        ViolinPlot(config=cng, data_dir=args.data).run()
