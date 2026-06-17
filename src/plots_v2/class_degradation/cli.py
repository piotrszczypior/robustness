import argparse
from pathlib import Path

from task import Task
from model import MODELS
from fragile.experiments import EXPERIMENTS, get_dfs_for_all_models
from .plot import render


TASK_NAME = "class_degradation_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="Class degradation plot v2 (clean vs corrupted accuracy per class)",
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


def _format_title(exp_name: str, model_label: str) -> str:
    return f"{model_label}: {exp_name.replace('_', ' ').capitalize()}"


def run(args: argparse.Namespace) -> None:
    out_base = Path(args.output_dir)

    for exp_name, variations in EXPERIMENTS.items():
        print(f"\n[class_degradation_v2] experiment: {exp_name}")
        dfs = get_dfs_for_all_models(variations, args.data_path)

        for model_key, model_label in MODELS.items():
            if model_key not in dfs:
                continue
            out = (
                out_base
                / "images"
                / "v3"
                / "class_degradation"
                / f"{exp_name}_{model_key}.png"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            render(dfs[model_key], out, title=_format_title(exp_name, model_label))
            print(f"  {exp_name}_{model_key}.png")
