import argparse
from pathlib import Path

from task import Task
from model import MODELS
from fragile.experiments import EXPERIMENTS, get_dfs_for_all_models
from .plot import render


TASK_NAME = "fragile_histogram_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="Relative drop histogram per model per experiment (P75+ highlighted)",
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


def run(args: argparse.Namespace) -> None:
    out_base = Path(args.output_dir)

    for exp_name, variations in EXPERIMENTS.items():
        print(f"\n[{TASK_NAME}] experiment: {exp_name}")
        dfs = get_dfs_for_all_models(variations, args.data_path)

        for model_key, model_label in MODELS.items():
            if model_key not in dfs:
                continue
            out = (
                out_base
                / "images"
                / "v2"
                / "fragile_histogram"
                / f"{exp_name}_{model_key}.png"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            title = f"{model_label}: {exp_name.replace('_', ' ').capitalize()}"
            render(dfs[model_key], out, title=title)
            print(f"  {exp_name}_{model_key}.png")
