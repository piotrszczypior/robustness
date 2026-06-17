import argparse
from pathlib import Path

import pandas as pd

from task import Task
from .plot import render


TASK_NAME = "corruption_taxonomy_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="Spearman rank correlation heatmap and dendrogram of corruption groups",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="results/representations/vit_b_16_class_metrics.parquet",
        help="Path to per-class metrics parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Base output directory",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="angular_distance_median",
        help="Metric column to use for correlation (default: angular_distance_median)",
    )


def run(args: argparse.Namespace) -> None:
    print(f"[{TASK_NAME}] metric: {args.metric}")
    df = pd.read_parquet(args.metrics_path)
    render(df, args.metric, Path(args.output_dir))
    print(f"[{TASK_NAME}] done")
