import argparse
from pathlib import Path

from task import Task
from model import MODELS
from fragile.experiments import EXPERIMENTS, get_dfs_for_all_models
from .plot import build_rmce_matrix, render


TASK_NAME = "rmce_heatmap_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(TASK_NAME, help="RmCE heatmap across models v2")
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


def run(args: argparse.Namespace) -> None:
    out_base = Path(args.output_dir)

    for exp_name, variations in EXPERIMENTS.items():
        print(f"\n[rmce_heatmap_v2] experiment: {exp_name}")
        dfs = get_dfs_for_all_models(variations, args.data_path)
        named = {MODELS[k]: v for k, v in dfs.items() if k in MODELS}
        matrix = build_rmce_matrix(named)
        out = out_base / "images" / "v2" / "rmce_heatmap" / f"{exp_name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        render(matrix, out)
        print(f"  {exp_name}.png")
