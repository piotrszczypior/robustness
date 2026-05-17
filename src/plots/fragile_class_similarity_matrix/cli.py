from __future__ import annotations

import argparse
import logging
from .settings import get_fragile_class_similarity_matrix_specs
from .plot import FragileClassSimilarityMatrix

__all__ = ["register", "run"]

PLOT_NAME = "fragile_similarity"

logger = logging.getLogger(PLOT_NAME)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser(PLOT_NAME, help="Generate fragile class similarity matrix plots")
    parser.add_argument("--files", nargs="+", help="List of JSON files to compare", required=True)
    parser.add_argument("--names", nargs="+", help="List of names for the files", required=True)
    parser.add_argument("--data", default="results/")
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(fragile_similarity_run=run)
    # fmt: on


def run(args: argparse.Namespace):
    if len(args.files) != len(args.names):
        raise ValueError("Number of files and names must be the same.")

    logger.info(f"Generating fragile similarity matrix for {args.files}")

    plot_spec = get_fragile_class_similarity_matrix_specs(
        files=args.files,
        names=args.names,
    )

    FragileClassSimilarityMatrix(config=plot_spec, data_dir=args.data).run()
