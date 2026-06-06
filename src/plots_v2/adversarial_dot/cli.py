import argparse
from pathlib import Path

import pandas as pd

from task import Task
from .plot import plot_adversarial_dot_plot, plot_adversarial_multi_class


TASK_NAME = "adversarial_dot_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="Cleveland dot plot — baseline vs adversarial accuracy per class",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="Adversarial CSV files to plot",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Base output directory",
    )


def run(args: argparse.Namespace) -> None:
    frames = [pd.read_csv(f) for f in args.files]
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        print(f"[adversarial_dot_v2] no data in provided files")
        return

    out_dir = Path(args.output_dir) / "images" / "adversarial" / "dot_plot"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(df["model"].unique())
    attacks = sorted(df["attack"].unique())

    filename = args.files[0].split("/")[-1]

    for model in models:
        sub = df[df["model"] == model]
        for attack in attacks:
            out = out_dir / f"{filename}_{attack}.png"
            plot_adversarial_multi_class(sub, attack, str(out))
            print(f"  {out.name}")
