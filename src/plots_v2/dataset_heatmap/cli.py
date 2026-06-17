import argparse
from pathlib import Path
from re import L

from space import CorruptionVariations
from task import Task
from model import MODELS
from fragile.experiments import (
    EXPERIMENTS,
    get_dfs_for_all_models,
    get_dfs_for_dataset,
    get_dfs_for_experiment,
)
from .plot import build_matrix, render


TASK_NAME = "dataset_heatmap"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME, help="Dataset accuracy heatmap across models"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="results",
        help="Path to per-class accuracy CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Base output directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet_c",
        # choices=["imagenet_a", "imagenet_r", "imagenet_c", "imagenet"],
        help="Dataset to visualize",
    )
    parser.add_argument(
        "--exp",
        type=str,
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default=None,
        help=(
            "Sort classes by accuracy. Use 'mean' for cross-model mean, "
            "a model key (e.g. 'resnet50') for a single model, "
            "or omit for original class index order (no sort)."
        ),
    )


def run(args: argparse.Namespace) -> None:
    out_base = Path(args.output_dir)

    # if args.dataset:
    #     print(f"[dataset_heatmap] loading data for {args.dataset}...")
    #     dfs = get_dfs_for_dataset(args.dataset, args.data_path)
    #     suffix = f"_sortby_{args.sort_by}" if args.sort_by else "_nosort"
    # else:
    experiment = EXPERIMENTS[args.exp]
    dfs = get_dfs_for_all_models(experiment, args.data_path)
    suffix = f"_sortby_{args.sort_by}" if args.sort_by else "_nosort"
    suffix = f"{suffix}_{args.exp}"

    named = {MODELS[k]: v for k, v in dfs.items() if k in MODELS}

    if args.sort_by == "mean":
        sort_by_label = "mean"
    elif args.sort_by:
        sort_by_label = MODELS.get(args.sort_by, args.sort_by)
    else:
        sort_by_label = None

    matrix = build_matrix(named, args.dataset, sort_by=sort_by_label)
    out = out_base / "images" / "v3" / "dataset_heatmap" / f"{args.dataset}_heatmap{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(matrix, out, args.dataset, sort_by=sort_by_label)
    print(f"  {out.name} ({matrix.shape[1]} classes, {matrix.shape[0]} models)")
