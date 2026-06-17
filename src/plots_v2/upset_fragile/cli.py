import argparse
from pathlib import Path

from task import Task
from space import CorruptionVariations
from constants import IMAGENET_C_SEVERITIES, IMAGENET_C_CORRUPTION_GROUPS
from fragile.experiments import get_dfs_for_experiment, get_df_for_model
from .plot import GROUPS, fragile_synsets, render


TASK_NAME = "upset_fragile_v2"

_MODELS = {
    "resnet50": "ResNet-50",
    "efficientnet_b4": "EfficientNet-B4",
    "vit_b_16": "ViT-B/16",
    "convnext_base": "ConvNeXt-Base",
}


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="UpSet plot of fragile class intersections (per corruption group or per corruption)",
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
        "--mode",
        choices=["groups", "corruptions", "severity", "groups_severity", "models"],
        default="groups",
        help="'groups': one plot per model (4 corruption groups as sets); "
             "'corruptions': one plot per model per group (individual corruptions as sets); "
             "'severity': one plot per model for a single corruption across severity levels; "
             "'groups_severity': like 'groups' but filtered to a single severity level; "
             "'models': one plot for a single corruption+severity across all models",
    )
    parser.add_argument(
        "--corruption",
        type=str,
        default="zoom_blur",
        help="Corruption name used in 'severity' mode (default: zoom_blur)",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=4,
        help="Severity level used in 'groups_severity' mode (default: 4)",
    )


def _run_by_groups(args: argparse.Namespace) -> None:
    out_base = Path(args.output_dir) / "images" / "v3" / "upset_fragile"

    for model_key, model_label in _MODELS.items():
        print(f"[{TASK_NAME}] {model_key}")
        group_sets: dict[str, set] = {}
        for group in GROUPS:
            df = get_dfs_for_experiment(group, model_key, args.data_path)
            group_sets[group] = fragile_synsets(df)

        out = out_base / f"{model_key}.png"
        render(group_sets, model_label, out, groups=GROUPS)
        print(f"  -> {out}")


def _run_by_corruptions(args: argparse.Namespace) -> None:
    out_base = Path(args.output_dir) / "images" / "v3" / "upset_fragile" / "per_corruption"

    for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items():
        for model_key, model_label in _MODELS.items():
            print(f"[{TASK_NAME}] {group} / {model_key}")
            corruption_sets: dict[str, set] = {}
            for corruption in corruptions:
                variations = CorruptionVariations(
                    groups=[group],
                    corruptions=[corruption],
                    severities=[4],
                )
                df = get_df_for_model(variations, model_key, args.data_path)
                corruption_sets[corruption] = fragile_synsets(df)

            title = f"{model_label} — {group.capitalize()}"
            out = out_base / group / f"{model_key}.png"
            render(corruption_sets, title, out, groups=corruptions)
            print(f"  -> {out}")


def _run_by_groups_severity(args: argparse.Namespace) -> None:
    severity = args.severity
    out_base = Path(args.output_dir) / "images" / "v3" / "upset_fragile" / f"groups_sev{severity}"

    for model_key, model_label in _MODELS.items():
        print(f"[{TASK_NAME}] {model_key} / severity {severity}")
        group_sets: dict[str, set] = {}
        for group in GROUPS:
            variations = CorruptionVariations(
                groups=[group],
                severities=[severity],
            )
            df = get_df_for_model(variations, model_key, args.data_path)
            group_sets[group] = fragile_synsets(df)

        out = out_base / f"{model_key}.png"
        render(group_sets, model_label, out, groups=GROUPS)
        print(f"  -> {out}")


def _run_by_severity(args: argparse.Namespace) -> None:
    corruption = args.corruption
    out_base = Path(args.output_dir) / "images" / "v3" / "upset_fragile" / "per_severity"
    severity_labels = [f"Severity {s}" for s in IMAGENET_C_SEVERITIES]

    for model_key, model_label in _MODELS.items():
        print(f"[{TASK_NAME}] {corruption} / {model_key}")
        severity_sets: dict[str, set] = {}
        for severity in IMAGENET_C_SEVERITIES:
            variations = CorruptionVariations(
                corruptions=[corruption],
                severities=[severity],
            )
            df = get_df_for_model(variations, model_key, args.data_path)
            severity_sets[f"Severity {severity}"] = fragile_synsets(df)

        title = f"{model_label} — {corruption.replace('_', ' ').capitalize()}"
        out = out_base / corruption / f"{model_key}.png"
        render(severity_sets, title, out, groups=severity_labels)
        print(f"  -> {out}")


def _run_by_models(args: argparse.Namespace) -> None:
    corruption = args.corruption
    severity = args.severity
    out_base = Path(args.output_dir) / "images" / "v3" / "upset_fragile" / "per_model"
    model_labels = list(_MODELS.values())

    print(f"[{TASK_NAME}] {corruption} / severity {severity}")
    model_sets: dict[str, set] = {}
    for model_key, model_label in _MODELS.items():
        variations = CorruptionVariations(
            corruptions=[corruption],
            severities=[severity],
        )
        df = get_df_for_model(variations, model_key, args.data_path)
        model_sets[model_label] = fragile_synsets(df)

    title = f"{corruption.replace('_', ' ').capitalize()} — Severity {severity}"
    out = out_base / f"{corruption}_sev{severity}.png"
    render(model_sets, title, out, groups=model_labels)
    print(f"  -> {out}")


def run(args: argparse.Namespace) -> None:
    if args.mode == "corruptions":
        _run_by_corruptions(args)
    elif args.mode == "severity":
        _run_by_severity(args)
    elif args.mode == "groups_severity":
        _run_by_groups_severity(args)
    elif args.mode == "models":
        _run_by_models(args)
    else:
        _run_by_groups(args)
