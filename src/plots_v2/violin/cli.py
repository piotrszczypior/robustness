import argparse
from pathlib import Path

from task import Task
from model import MODELS
from fragile.experiments import EXPERIMENTS, get_dfs_for_all_models
from .plot import render


TASK_NAME = "violin_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(TASK_NAME, help="Violin plots v2")
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
        dfs = get_dfs_for_all_models(variations, args.data_path)

        for model, extern in MODELS.items():
            out = out_base / "images" / "v3" / "violin" / f"{exp_name}_{model}.png"
            df_named = {extern: dfs[model]}
            out.parent.mkdir(parents=True, exist_ok=True)
            render(df_named, out, title=exp_name.replace("_", " ").capitalize())

    print("FINISH")
