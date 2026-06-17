import argparse
from pathlib import Path

from task import Task
from model import MODELS
from fragile.experiments import EXPERIMENTS, get_dfs_for_all_models
from .plot import (
    compute_c_sets,
    compute_jaccard_matrix,
    compute_fisher_matrix,
    render_jaccard,
    render_fisher,
)


TASK_NAME = "rmce_overlap_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="Fisher exact test + Jaccard heatmaps of the criterion-C (RmCE) "
        "fragile set across models",
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
    parser.add_argument(
        "--exp",
        type=str,
        default="all_corruptions",
        choices=list(EXPERIMENTS.keys()),
        help="Experiment name (default: all_corruptions)",
    )


def run(args: argparse.Namespace) -> None:
    print(f"\n[{TASK_NAME}] experiment: {args.exp}")

    variations = EXPERIMENTS[args.exp]
    raw_dfs = get_dfs_for_all_models(variations, args.data_path)

    c_sets, universes = compute_c_sets(raw_dfs)

    # Label by human-readable model name, keeping MODELS ordering.
    model_keys = [k for k in MODELS.keys() if k in raw_dfs]
    labels = [MODELS[k] for k in model_keys]
    c_sets = {MODELS[k]: c_sets[k] for k in model_keys}
    universes = {MODELS[k]: universes[k] for k in model_keys}

    jaccard = compute_jaccard_matrix(c_sets, labels)
    p_matrix = compute_fisher_matrix(c_sets, universes, labels)

    out_base = Path(args.output_dir) / "images" / "v3" / "rmce_overlap"
    title = f"Criterion C — {args.exp.replace('_', ' ').title()}"

    jaccard_out = out_base / f"jaccard_{args.exp}.png"
    jaccard_out.parent.mkdir(parents=True, exist_ok=True)
    render_jaccard(jaccard, jaccard_out, title=f"Jaccard: {title}")
    print(f"  Saved: {jaccard_out.with_suffix('.pdf')}")

    fisher_out = out_base / f"fisher_{args.exp}.png"
    fisher_out.parent.mkdir(parents=True, exist_ok=True)
    render_fisher(p_matrix, fisher_out, title=f"Fisher Test: {title}")
    print(f"  Saved: {fisher_out.with_suffix('.pdf')}")
