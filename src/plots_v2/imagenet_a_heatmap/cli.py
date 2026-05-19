import argparse
from pathlib import Path

from task import Task
from model import MODELS
from fragile.experiments import get_dfs_for_dataset
from .plot import build_matrix, render


TASK_NAME = "imagenet_a_heatmap_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME, help="ImageNet-A accuracy heatmap across models v2"
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
        default="imagenet_a",
        choices=["imagenet_a", "imagenet_r"],
        help="Dataset to visualize",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default=None,
        help="Sort classes by a single model's accuracy (e.g. 'resnet50'). Default: mean across all models.",
    )


def run(args: argparse.Namespace) -> None:
    out_base = Path(args.output_dir)

    print(f"[imagenet_a_heatmap_v2] loading data for {args.dataset}...")
    dfs = get_dfs_for_dataset(args.dataset, args.data_path)
    named = {MODELS[k]: v for k, v in dfs.items() if k in MODELS}

    sort_by_label = MODELS.get(args.sort_by) if args.sort_by else None

    matrix = build_matrix(named, args.dataset, sort_by=sort_by_label)
    suffix = f"_sortby_{args.sort_by}" if args.sort_by else ""
    out = out_base / "images" / "v2" / f"{args.dataset}_heatmap{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(matrix, out, args.dataset)
    print(f"  {out.name} ({matrix.shape[1]} classes, {matrix.shape[0]} models)")
