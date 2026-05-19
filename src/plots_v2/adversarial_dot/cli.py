import argparse
from pathlib import Path

from task import Task
from .plot import load_adversarial_results, plot_adversarial_dot_plot


TASK_NAME = "adversarial_dot_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME, help="Cleveland dot plot — baseline vs adversarial accuracy per class"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="aversarial",
        help="Directory with adversarial CSV files (default: aversarial)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Base output directory",
    )


def run(args: argparse.Namespace) -> None:
    df = load_adversarial_results(args.data_path)
    if df.empty:
        print(f"[adversarial_dot_v2] no CSV files found in {args.data_path}")
        return

    out_dir = Path(args.output_dir) / "images" / "adversarial" / "dot_plot"
    out_dir.mkdir(parents=True, exist_ok=True)

    synsets = sorted(df["synset"].unique())
    models = sorted(df["model"].unique())
    attacks = sorted(df["attack"].unique())

    for synset in synsets:
        for model in models:
            for attack in attacks:
                sub = df[
                    (df["synset"] == synset)
                    & (df["model"] == model)
                    & (df["attack"] == attack)
                ]
                if sub.empty:
                    continue
                out = out_dir / f"{synset}_{model}_{attack}.png"
                plot_adversarial_dot_plot(sub, attack, str(out))
                print(f"  {out.name}")
