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

    from .jaccard_overlap.cli import register as register_jaccard
    register_jaccard(plot_subparsers)

    from .violin_plot.cli import register as register_violin
    register_violin(plot_subparsers)

    from .violin_plot_rmce import register as register_violin_rmce
    register_violin_rmce(plot_subparsers)

    from .barcode_plot_rmce import register as register_barcode_plot_rmce
    register_barcode_plot_rmce(plot_subparsers)

    from .accuracy_scatter_plot.cli import register as register_accuracy_scatter
    register_accuracy_scatter(plot_subparsers)

    from .class_degradation_plot.cli import register as register_class_degradation
    register_class_degradation(plot_subparsers)

    from .barcode_fragile_classes.cli import register as register_barcode_fragile_classes
    register_barcode_fragile_classes(plot_subparsers)

    from .fragile_class_similarity_matrix.cli import register as register_fragile_similarity
    register_fragile_similarity(plot_subparsers)

    from .spearman_rank_plot.cli import register as register_spearman_rank
    register_spearman_rank(plot_subparsers)
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

    if args.plot_command == "jaccard":
        args.jaccard_run(args)
        return

    if args.plot_command == "violin":
        args.violin_run(args)
        return

    if args.plot_command == "violin_rmce":
        args.violin_rmce_run(args)
        return

    if args.plot_command == "barcode_rmce":
        args.barcode_rmce_run(args)
        return

    if args.plot_command == "accuracy_scatter":
        args.accuracy_scatter_run(args)
        return

    if args.plot_command == "class_degradation":
        args.class_degradation_run(args)
        return

    if args.plot_command == "barcode_fragile_classes":
        args.barcode_fragile_classes_run(args)
        return

    if args.plot_command == "fragile_similarity":
        args.fragile_similarity_run(args)
        return

    if args.plot_command == "spearman_rank":
        args.spearman_rank_run(args)
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
