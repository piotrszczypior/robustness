import argparse
from pathlib import Path

from task import Task
from model import MODELS
from fragile.experiments import EXPERIMENTS, get_dfs_for_all_models
from .plot import compute_spearman, render


TASK_NAME = "spearman_v2"

METRICS = ["rel_drop", "abs_drop", "mCE", "RmCE", "nCE"]

# RmCE and mCE are defined relative to AlexNet, so AlexNet's per-class values
# are a trivial constant (1.0) — exclude it to avoid degenerate correlation.
_ALEXNET_RELATIVE_METRICS = {"RmCE", "mCE"}


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(TASK_NAME, help="Spearman correlation heatmaps v2")
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
        print(f"\n[spearman_v2] experiment: {exp_name}")
        raw_dfs = get_dfs_for_all_models(variations, args.data_path)

        dfs = {MODELS[k]: v for k, v in raw_dfs.items() if k in MODELS}

        for metric in METRICS:
            metric_dfs = {
                name: df for name, df in dfs.items()
                if not (metric in _ALEXNET_RELATIVE_METRICS and name == "AlexNet")
            }
            corr = compute_spearman(metric_dfs, metric)
            out_path = out_base / "images" / "spearman" / "v2" / metric / f"{exp_name}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            render(corr, out_path)
            print(f"  {metric}/{exp_name}.png")
