import argparse
from pathlib import Path

from task import Task
from model import MODELS
from space import CorruptionVariations
from fragile.experiments import get_dfs_for_all_models, get_df_for_model
from constants import IMAGENET_C_SEVERITIES, IMAGENET_C_CORRUPTION_GROUPS
from .plot import render, select_synsets, select_synsets_single_model


TASK_NAME = "severity_dot_v2"

_STANDARD_GROUPS = [k for k in IMAGENET_C_CORRUPTION_GROUPS if k != "extra"]


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="Dot plot of per-class accuracy across corruption severities",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="results",
        help="Path to per-class accuracy CSV files (default: results)",
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
    )
    parser.add_argument(
        "--model",
        type=str,
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=_STANDARD_GROUPS,
        choices=_STANDARD_GROUPS,
        help="Corruption groups to plot",
    )


def run(args: argparse.Namespace) -> None:
    out_base = Path(args.output_dir)

    for group in args.groups:
        severity_dfs = {}
        for sev in IMAGENET_C_SEVERITIES:
            variations = CorruptionVariations(groups=[group], severities=[sev])
            severity_dfs[sev] = get_df_for_model(variations, args.model, args.data_path)

        selected_synsets = select_synsets_single_model(severity_dfs)
        model = args.model
        out = out_base / "images" / "v2" / "severity_dot" / f"{group}_{model}.png"
        render(severity_dfs, model, group.capitalize(), out, selected_synsets)
        print(f"{group}_{model}.png")

    # for group in args.groups:
    #     print(f"\n[severity_dot_v2] group: {group}")
    #     severity_dfs = {}
    #     for sev in IMAGENET_C_SEVERITIES:
    #         variations = CorruptionVariations(groups=[group], severities=[sev])
    #         severity_dfs[sev] = get_dfs_for_all_models(variations, args.data_path)

    #     selected_synsets = select_synsets(severity_dfs)

    #     models = args.models if args.models else list(MODELS.keys())
    #     for model_key in models:
    #         if model_key not in severity_dfs[1]:
    #             continue
    #         model_label = MODELS.get(model_key, model_key)
    #         out = out_base / "images" / "v2" / "severity_dot" / f"{group}_{model_key}.png"
    #         render(severity_dfs, model_key, group.capitalize(), out, selected_synsets)
    #         print(f"  {group}_{model_key}.png")
