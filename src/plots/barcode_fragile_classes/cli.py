from __future__ import annotations

import argparse
import logging
from .settings import get_barcode_fragile_classes_specs
from .plot import BarcodeFragileClassesFreq

__all__ = ["register", "run"]

PLOT_NAME = "barcode_fragile_classes"

logger = logging.getLogger(PLOT_NAME)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser(PLOT_NAME, help="Generate barcode plots for fragile classes")
    parser.add_argument("--data", default="results/")
    parser.add_argument("--group", type=str)
    parser.add_argument("--models", nargs="+", type=str, help="Specific models to plot")
    parser.add_argument("--corruptions", nargs="+", type=str, help="Specific corruptions to plot")
    parser.add_argument("--severities", nargs="+", type=int, help="Specific severities to plot")
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(barcode_fragile_classes_run=run)
    # fmt: on


def run(args: argparse.Namespace):
    logger.info("Generating barcode fragile classes plots")

    plots = list(
        get_barcode_fragile_classes_specs(
            models=args.models,
            corruptions=args.corruptions,
            severities=args.severities,
            group=args.group,
        )
    )

    for cng in plots:
        BarcodeFragileClassesFreq(config=cng, data_dir=args.data).run()
