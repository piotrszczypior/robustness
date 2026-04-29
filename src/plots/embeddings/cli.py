from __future__ import annotations

import argparse
import logging
from plots.specs import get_plot_specs
from .embeddings import EmbeddingProjector

__all__ = ["register"]

PLOT_NAME = "embeddings"

logger = logging.getLogger(PLOT_NAME)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser(PLOT_NAME, help="Generate embedding projections")
    parser.add_argument("--settings", default="src/plots/embeddings/settings.yaml")
    parser.add_argument("--data", default="results/")
    parser.add_argument("--projection", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(embeddings_run=run)
    # fmt: on


def run(args: argparse.Namespace):
    logger.info(args.settings)

    plots = get_plot_specs(args.settings)
    for cng in plots:
        EmbeddingProjector(config=cng, data_dir=args.data).run()
