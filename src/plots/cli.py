from __future__ import annotations

import argparse
import logging

from task import Task
from .recipe import register_recipes
from .specs import get_plot_specs
from .analyze import create_plot

__all__ = ["get_task", "register", "run"]

TASK_NAME = "plot"

logger = logging.getLogger(TASK_NAME)


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser("plot", help="Generate analysis plots from results")
    parser.add_argument("--plots", default="plots/plots.yaml", type=str, help="Plots configuration file")
    parser.add_argument("--recipes", default="plots/recipes.yaml", type=str, help="Recipes configuration file")
    parser.add_argument("--data", default="results/", type=str, help="Data directory with csv files")
    parser.add_argument("--sync-drive", action="store_true", help="Sync results to Google Drive")
    # fmt: on


def run(args: argparse.Namespace):
    logger.info("Starting Plotting Task")

    logger.info(f"Loading recipes from: {args.recipes}")
    register_recipes(args.recipes)

    logger.info(f"Loading plot specs from: {args.plots}")
    plots = list(get_plot_specs(args.plots))
    total_plots = len(plots)

    if len(plots) == 0:
        logger.warning("Execution halted: No valid plot configurations found.")
        return

    logger.info(f"Orchestrating {total_plots} visualization task(s).")

    for i, plot in enumerate(plots, 1):
        logger.info(f"[{i}/{total_plots}] Rendering '{plot.name}'...")
        try:
            create_plot(plot, args.data)
        except Exception as e:
            logger.error(f"  FAILED rendering '{plot.name}': {e}")

    logger.info("All Plotting Tasks Completed Successfully ---")
