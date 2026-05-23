import argparse
from pathlib import Path

from task import Task
from space import CorruptionVariations
from constants import IMAGENET_C_SEVERITIES
from fragile.experiments import get_df_for_model
from fragile.fragile import get_absolute_fragile, get_relative_drop_fragile
from .plot import render


TASK_NAME = "fragile_severity_line_v2"

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
        help="Line plot of fragile class count across severity levels per model",
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
        "--corruption",
        type=str,
        default="zoom_blur",
        help="Corruption name to plot (default: zoom_blur)",
    )


def _fragile_count(df) -> int:
    df = get_absolute_fragile(df)
    df = get_relative_drop_fragile(df)
    return int(((df["is_fragile_a"] == 1) & (df["is_fragile_b"] == 1)).sum())


def run(args: argparse.Namespace) -> None:
    corruption = args.corruption
    out_base = Path(args.output_dir) / "images" / "v2" / "fragile_severity_line"

    print(f"[{TASK_NAME}] {corruption}")
    data: dict[str, list[int]] = {}
    for model_key, model_label in _MODELS.items():
        counts = []
        for severity in IMAGENET_C_SEVERITIES:
            variations = CorruptionVariations(
                corruptions=[corruption],
                severities=[severity],
            )
            df = get_df_for_model(variations, model_key, args.data_path)
            counts.append(_fragile_count(df))
        data[model_label] = counts

    title = corruption.replace("_", " ").capitalize()
    out = out_base / f"{corruption}.png"
    render(data, title, out)
    print(f"  -> {out}")
