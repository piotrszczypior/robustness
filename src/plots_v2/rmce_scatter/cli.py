import argparse
from pathlib import Path

from task import Task
from model import MODELS
from fragile.experiments import EXPERIMENTS, get_dfs_for_all_models
from .plot import render


TASK_NAME = "rmce_scatter_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(TASK_NAME, help="RmCE scatter plot per model")
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
        "--models",
        nargs="+",
        default=None,
        help="Model keys to plot (default: all)",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Single experiment name (default: all experiments)",
    )


def run(args: argparse.Namespace) -> None:
    out_base = Path(args.output_dir)
    experiments = (
        {args.exp: EXPERIMENTS[args.exp]} if args.exp else EXPERIMENTS
    )
    model_keys = args.models

    for exp_name, variations in experiments.items():
        print(f"\n[rmce_scatter_v2] experiment: {exp_name}")
        dfs = get_dfs_for_all_models(variations, args.data_path)

        for model_key, df in dfs.items():
            if model_keys and model_key not in model_keys:
                continue
            if "RmCE" not in df.columns or df["RmCE"].isna().all():
                print(f"  {model_key}: no RmCE data, skipping")
                continue
            model_label = MODELS.get(model_key, model_key)
            out = out_base / "images" / "v2" / "rmce_scatter" / f"{exp_name}_{model_key}.png"
            render(df, model_label, out)
            print(f"  {exp_name}_{model_key}.png")
