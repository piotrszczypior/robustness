from __future__ import annotations

import argparse
import logging

from paths import paths
from task import Task

from . import dataset, runner
from .loader import list_conditions
from .pca_scatter import run_pca_scatter

__all__ = ["get_task", "register", "run"]

TASK_NAME = "representations"

logger = logging.getLogger(TASK_NAME)


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME, help="Analyze representation vectors across corruptions"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=str(paths.embeddings),
        help="Directory with *_embeddings.npy / .parquet pairs",
    )

    commands = parser.add_subparsers(dest="representations_commands", required=True)

    list_parser = commands.add_parser("list", help="List available embedding pairs")
    list_parser.set_defaults(representations_func=run_list)

    run_parser = commands.add_parser(
        "run",
        help="Analyze a model across its corruption space (clean is auto-derived)",
    )
    run_parser.add_argument(
        "--model", default="resnet50", help="Model whose embeddings to analyze"
    )
    run_parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Corruption groups to include (default: all)",
    )
    run_parser.add_argument(
        "--corruptions",
        nargs="+",
        default=None,
        help="Corruptions to include (default: all in selected groups)",
    )
    run_parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=None,
        help="Severities to include (default: 1 2 3 4 5)",
    )
    run_parser.set_defaults(representations_func=run_analysis)

    rollup_parser = commands.add_parser(
        "roll-up",
        help="Roll 1.1-1.3 over all conditions into one tidy table and save it",
    )
    rollup_parser.add_argument(
        "--model", default="resnet50", help="Model whose embeddings to analyze"
    )
    rollup_parser.add_argument(
        "--groups", nargs="+", default=None, help="Corruption groups (default: all)"
    )
    rollup_parser.add_argument(
        "--corruptions", nargs="+", default=None, help="Corruptions (default: all)"
    )
    rollup_parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=None,
        help="Severities (default: 1 2 3 4 5)",
    )
    rollup_parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: results/representations/{model}_class_metrics.parquet)",
    )
    rollup_parser.set_defaults(representations_func=run_rollup)

    pca_parser = commands.add_parser(
        "pca-scatter",
        help="PCA scatter of raw embeddings for selected synsets vs a single corruption",
    )
    pca_parser.add_argument(
        "--model", default="resnet50", help="Model whose embeddings to use"
    )
    pca_parser.add_argument(
        "--synsets",
        required=True,
        help="Comma-separated synset IDs, e.g. n01773157,n01774384",
    )
    pca_parser.add_argument("--corruption", required=True)
    pca_parser.add_argument("--severity", type=int, required=True)
    pca_parser.add_argument(
        "--n-samples", type=int, default=50, dest="n_samples",
        help="Samples per synset per condition (default: 50)",
    )
    pca_parser.add_argument("--seed", type=int, default=42)
    pca_parser.add_argument(
        "--out", default=None,
        help="Output PNG path (default: images/representations/pca/{model}_{corruption}_{severity}.png)",
    )
    pca_parser.set_defaults(representations_func=run_pca_scatter)


def run(args: argparse.Namespace) -> None:
    args.representations_func(args)


def run_list(args: argparse.Namespace) -> None:
    conditions = list_conditions(args.embeddings_dir)
    logger.info("Found %d embedding pairs in %s", len(conditions), args.embeddings_dir)
    for name in conditions:
        print(name)


def run_analysis(args: argparse.Namespace) -> None:
    runner.run(
        model=args.model,
        groups=args.groups,
        corruptions=args.corruptions,
        severities=args.severities,
        embeddings_dir=args.embeddings_dir,
    )


def run_rollup(args: argparse.Namespace) -> None:
    tidy = dataset.build_tidy(
        model=args.model,
        groups=args.groups,
        corruptions=args.corruptions,
        severities=args.severities,
        embeddings_dir=args.embeddings_dir,
    )

    out = args.out or str(
        paths.results / "representations" / f"{args.model}_class_metrics.parquet"
    )
    path = dataset.save_tidy(tidy, out)

    print(f"model      : {args.model}")
    print(f"rows       : {len(tidy)}")
    print(f"conditions : {tidy[['corruption', 'severity']].drop_duplicates().shape[0]}")
    print(f"synsets    : {tidy['synset'].nunique()}")
    print(f"metrics    : {sorted(tidy['metric'].unique())}")
    print(f"saved      : {path}")
    print("\nhead:")
    print(tidy.head(8).to_string(index=False))
