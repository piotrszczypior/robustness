import argparse
from pathlib import Path

from task import Task
from model import MODELS
from space import CorruptionVariations
from fragile.experiments import EXPERIMENTS, get_dfs_for_all_models
from fragile.definitions import DEFINITIONS
from .plot import compute_jaccard, render


TASK_NAME = "jaccard_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(TASK_NAME, help="Jaccard Index heatmaps for fragile classes v2")
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

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--exp",
        type=str,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment name (aggregated across corruptions/severities)",
    )
    group.add_argument(
        "--corruption",
        type=str,
        help="Single corruption type, e.g. defocus_blur",
    )

    parser.add_argument(
        "--severity",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Severity level (required when --corruption is used)",
    )


def run(args: argparse.Namespace) -> None:
    if args.exp:
        variations = EXPERIMENTS[args.exp]
        condition = args.exp
    else:
        if args.severity is None:
            raise ValueError("--severity is required when --corruption is used")
        variations = CorruptionVariations(
            corruptions=[args.corruption],
            severities=[args.severity],
        )
        condition = f"{args.corruption}_{args.severity}"

    print(f"\n[{TASK_NAME}] condition: {condition}")
    raw_dfs = get_dfs_for_all_models(variations, args.data_path)
    dfs = {MODELS[k]: v for k, v in raw_dfs.items() if k in MODELS}

    out_base = Path(args.output_dir)

    def_name = "ab"
    definition = DEFINITIONS[def_name]

    jaccard = compute_jaccard(dfs, definition)
    out_path = out_base / "images" / "v2" / "jaccard" / def_name / f"{condition}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render(jaccard, definition, out_path)
    print(f"  {def_name}/{condition}.png")
