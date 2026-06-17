import argparse
from pathlib import Path

import pandas as pd

from task import Task
from space import CorruptionVariations
from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES
from fragile.experiments import _build_df_per_setting
from .plot import compute_spearman_grid, render


TASK_NAME = "repr_drop_spearman_v2"

_STANDARD_GROUPS = [k for k in IMAGENET_C_CORRUPTION_GROUPS if k != "extra"]


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="Heatmap of Spearman correlation between per-class cosine distance "
        "(representation shift) and per-class accuracy drop, across corruption × severity",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Model whose representation metrics and accuracy are used",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Per-class metrics parquet (default: "
        "results/representations/{model}_class_metrics.parquet)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="results",
        help="Path to per-class accuracy CSV files (default: results)",
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
        help="Cosine-distance metric column (default: angular_distance_median)",
    )
    parser.add_argument(
        "--drop",
        type=str,
        default="abs_drop",
        choices=["abs_drop", "rel_drop"],
        help="Accuracy-drop measure to correlate (default: abs_drop)",
    )


def run(args: argparse.Namespace) -> None:
    metrics_path = (
        args.metrics_path
        or f"results/representations/{args.model}_class_metrics.parquet"
    )
    print(f"[{TASK_NAME}] model={args.model} metric={args.metric} drop={args.drop}")

    metrics_df = pd.read_parquet(metrics_path)

    variations = CorruptionVariations(
        groups=_STANDARD_GROUPS, severities=IMAGENET_C_SEVERITIES
    )
    drop_dfs = _build_df_per_setting(args.model, variations, args.data_path)

    grid = compute_spearman_grid(metrics_df, drop_dfs, args.metric, args.drop)

    out = (
        Path(args.output_dir)
        / "images"
        / "v3"
        / "repr_drop_spearman"
        / f"{args.model}_{args.metric}_{args.drop}.png"
    )
    render(grid, args.model, args.metric, args.drop, out)
    print(f"[{TASK_NAME}] done -> {out.with_suffix('.pdf')}")
