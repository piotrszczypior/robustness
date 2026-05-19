import argparse
from pathlib import Path

from task import Task
from model import MODELS
from fragile.experiments import EXPERIMENTS, get_dfs_for_all_models
from .plot import plot


TASK_NAME = "acc_to_acc_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME, help="Accuracy vs accuracy scatter plots v2"
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

    for name, experiment in EXPERIMENTS.items():
        dfs = get_dfs_for_all_models(experiment)

        for model, model_label in MODELS.items():
            out = out_base / "images" / "v2" / "acc_to_acc" / f"{name}_{model}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plot(dfs[model], out, title=format_title(name, model_label))
            print(f"  {name}: {model}")


def format_title(title: str, model: str):
    title = title.replace("_", " ").capitalize()

    return f"{model}: {title}"
