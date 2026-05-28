import argparse
from pathlib import Path

from task import Task
from model import MODELS
from fragile.experiments import EXPERIMENTS, get_dfs_for_all_models
from fragile.definitions import DEFINITIONS
from .plot import compute_fisher_matrix, render


TASK_NAME = "fisher_heatmap"


def _format_corruption_title(corruption: str, severity: int) -> str:
    human = corruption.replace("_", " ").title()
    return f"{human} Severity {severity}"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(TASK_NAME, help="Fisher exact test p-value heatmaps")
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
        default=None,
        help="Experiment name (e.g. blur, noise). If not set, runs all experiments.",
    )
    parser.add_argument(
        "--corruption",
        type=str,
        default=None,
        help="Specific corruption (e.g. defocus_blur)",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=None,
        help="Severity level (1-5)",
    )
    parser.add_argument(
        "--definition",
        type=str,
        default="ab",
        choices=list(DEFINITIONS.keys()),
        help="Fragile class definition",
    )


def run(args: argparse.Namespace) -> None:
    from space import CorruptionVariations
    from constants import IMAGENET_C_CORRUPTION_GROUPS

    out_base = Path(args.output_dir)

    if args.corruption and args.severity:
        group = None
        for g, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items():
            if args.corruption in corruptions:
                group = g
                break
        if not group:
            raise ValueError(f"Unknown corruption: {args.corruption}")

        variations = CorruptionVariations(
            groups=[group],
            corruptions=[args.corruption],
            severities=[args.severity],
        )
        exp_name = f"{args.corruption}_{args.severity}"

        print(f"\n[fisher_heatmap] {exp_name}, definition={args.definition}")
        dfs = get_dfs_for_all_models(variations, args.data_path)
        p_matrix = compute_fisher_matrix(dfs, args.definition)

        out_path = out_base / "images" / "fisher" / args.definition / f"{exp_name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        render(p_matrix, out_path, title=f"Fisher Test: {_format_corruption_title(args.corruption, args.severity)}")
        print(f"  Saved: {out_path}")

    else:
        experiments = {args.exp: EXPERIMENTS[args.exp]} if args.exp else EXPERIMENTS

        for exp_name, variations in experiments.items():
            print(f"\n[fisher_heatmap] experiment: {exp_name}")
            dfs = get_dfs_for_all_models(variations, args.data_path)
            p_matrix = compute_fisher_matrix(dfs, args.definition)

            out_path = (
                out_base / "images" / "fisher" / args.definition / f"{exp_name}.png"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            render(p_matrix, out_path, title=f"Fisher Test: {exp_name.replace('_', ' ').title()}")
            print(f"  Saved: {out_path}")
