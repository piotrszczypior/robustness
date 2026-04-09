from __future__ import annotations

import argparse
import logging

from analyze.specs import get_specs
from task import Task
from .core import run_analysis

logger = logging.getLogger(__name__)

TASK_NAME = "analyze"


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    # fmt: off
    parser = subparsers.add_parser("analyze", help="Analyze results")
    parser.add_argument("--path", default="analysis/base.yaml", type=str, help="Analysis configuration file")
    parser.add_argument("--data", default="results/", type=str, help="Data directory with csv files")
    parser.add_argument("--output", default="analysis_results/", type=str, help="Data directory with csv files")
    parser.add_argument("--sync-drive", action="store_true", help="Sync results to Google Drive")
    parser.add_argument("--debug", action="store_true", help="Skip plot generation")
    # fmt: on


def run(args: argparse.Namespace):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Loading analysis config from: {args.path}")
    specs = list(get_specs(args.path))
    total = len(specs)

    if total == 0:
        logger.warning("No valid configurations found.")
        return

    for i, spec in enumerate(specs, 1):
        logger.info(f"[{i}/{total}] Analysing '{spec.name}'...")
        run_analysis(spec, args.output)
