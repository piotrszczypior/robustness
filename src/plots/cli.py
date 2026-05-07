from __future__ import annotations

import argparse
import logging

from task import Task
from .specs import get_plot_specs
from .core import create_plot

__all__ = ["get_task", "register", "run"]

TASK_NAME = "plot"

logger = logging.getLogger(TASK_NAME)


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser("plot", help="Generate analysis plots from results")
    parser.add_argument("--plots", default="plots/plots.yaml", type=str, help="Plots configuration file")
    parser.add_argument("--data", default="results/", type=str, help="Data directory with csv files")
    parser.add_argument("--sync-drive", action="store_true", help="Sync results to Google Drive")
    parser.add_argument("--debug", action="store_true", help="Skip plot generation")

    plot_subparsers = parser.add_subparsers(dest="plot_command")

    from .embeddings.cli import register as register_embeddings
    register_embeddings(plot_subparsers)

    from .spearman_corr.cli import register as register_spearman
    register_spearman(plot_subparsers)
    # fmt: on


def run(args: argparse.Namespace):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting Plotting Task")

    if args.plot_command == "embeddings":
        args.embeddings_run(args)
        return

    if args.plot_command == "spearman":
        args.spearman_run(args)
        return

    logger.info(f"Loading plot specs from: {args.plots}")
    plots = list(get_plot_specs(args.plots))
    total_plots = len(plots)

    if total_plots == 0:
        logger.warning("Execution halted: No valid plot configurations found.")
        return

    logger.info(f"Orchestrating {total_plots} visualization task(s).")

    for i, plot in enumerate(plots, 1):
        logger.info(f"[{i}/{total_plots}] Rendering '{plot.name}'...")
        try:
            create_plot(plot, args.data, debug=args.debug)
        except Exception as e:
            logger.error(f"[ERROR] rendering '{plot.name}': {e}")
            raise RuntimeError(
                f"Plotting task failed due to error in '{plot.name}'"
            ) from e
