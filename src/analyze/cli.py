from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pandas as pd

from task import Task
from .analyses import get_settings
from .core import run_analysis
from .fragile import (
    ThresholdFilter,
    TailFilter,
    AccuracyDropFilter,
    find_overlapping_fragile_classes,
    export_to_json,
    export_to_latex,
)

logger = logging.getLogger(__name__)

TASK_NAME = "analyze"


def get_task():
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    # Base parser for analyze
    parser = subparsers.add_parser("analyze", help="Analyze results")

    # We add subparsers for "fragile"
    analyze_subparsers = parser.add_subparsers(dest="analyze_command")

    # Command: fragile
    fragile_parser = analyze_subparsers.add_parser(
        "fragile", help="Run fragile class analysis"
    )
    fragile_parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline CSV file name (e.g., resnet152_imagenet.csv)",
    )
    fragile_parser.add_argument(
        "--corrupted",
        type=str,
        nargs="+",
        required=True,
        help="Corrupted CSV file name(s)",
    )
    fragile_parser.add_argument(
        "--type",
        type=str,
        required=True,
        help="Filter strategy, e.g., '80:50', 'tail:25:worst', 'drop:15'",
    )
    fragile_parser.add_argument(
        "--to-latex", action="store_true", help="Generate LaTeX table"
    )
    fragile_parser.add_argument(
        "--output", type=str, default="output/fragile", help="Output directory override"
    )
    fragile_parser.add_argument(
        "--data", default="results/", type=str, help="Data directory with csv files"
    )

    # Arguments directly on 'analyze' for backwards compatibility
    parser.add_argument("--type", type=str, help="Filter by task type")
    parser.add_argument(
        "--data", default="results/", type=str, help="Data directory with csv files"
    )
    parser.add_argument(
        "--output", default="analysis/", type=str, help="Output directory"
    )
    parser.add_argument(
        "--sync-drive", action="store_true", help="Sync results to Google Drive"
    )
    parser.add_argument("--debug", action="store_true", help="Skip plot generation")


def parse_strategy(type_str: str):
    parts = type_str.split(":")
    if parts[0] == "tail" and len(parts) >= 2:
        k = int(parts[1])
        sort_by = parts[2] if len(parts) > 2 else "worst"
        return TailFilter(k=k, sort_by=sort_by), f"tail_{k}_{sort_by}"
    elif parts[0] == "drop" and len(parts) == 2:
        k = int(parts[1])
        return AccuracyDropFilter(k=k), f"drop_{k}"
    elif len(parts) == 2:
        # Assuming format 80:50
        clean_min = float(parts[0]) / 100.0 if float(parts[0]) > 1 else float(parts[0])
        corrupt_max = (
            float(parts[1]) / 100.0 if float(parts[1]) > 1 else float(parts[1])
        )
        return ThresholdFilter(
            clean_min=clean_min, corrupt_max=corrupt_max
        ), f"threshold_{clean_min}_{corrupt_max}"
    else:
        raise ValueError(f"Unknown fragile strategy type format: {type_str}")


def run_fragile(args: argparse.Namespace):
    logger.info(
        f"Running standalone fragile classes analysis with strategy: {args.type}"
    )

    data_dir = Path(args.data)
    baseline_path = data_dir / args.baseline
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    baseline_df = pd.read_csv(baseline_path)

    domain_pairs = []
    for corr_file in args.corrupted:
        corr_path = data_dir / corr_file
        if not corr_path.exists():
            raise FileNotFoundError(f"Corrupted file not found: {corr_path}")
        corr_df = pd.read_csv(corr_path)
        domain_pairs.append((baseline_df, corr_df))

    strategy, strat_name = parse_strategy(args.type)

    fragile_classes = find_overlapping_fragile_classes(domain_pairs, strategy)

    output_dir = Path(args.output) / strat_name
    export_to_json(fragile_classes, output_dir, "classes.json")

    if args.to_latex:
        latex = export_to_latex(fragile_classes)
        print("\n--- LaTeX Table ---")
        print(latex)
        print("-------------------\n")


def run(args: argparse.Namespace):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if hasattr(args, "analyze_command") and args.analyze_command == "fragile":
        run_fragile(args)
        return

    logger.info("Loading tasks")
    tasks = [t for t in get_settings() if not args.type or t.type == args.type]
    total = len(tasks)

    if total == 0:
        logger.warning("No valid configurations found.")
        return

    logger.info(f"Found {total} tasks")
    for i, spec in enumerate(tasks, 1):
        logger.info(f"[{i}/{total}] Analysing '{spec.name}'...")
        run_analysis(spec, args.output)
