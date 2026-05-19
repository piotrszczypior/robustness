import argparse
from pathlib import Path

from task import Task
from model import MODELS
from fragile.experiments import EXPERIMENTS, get_dfs_for_all_models, get_rmce_alexnet_df
from fragile.fragile import (
    get_absolute_fragile,
    get_relative_drop_fragile,
    get_rmce_fragile,
)
from fragile.definitions import DEFINITIONS
from .plot import build_barcode_matrix, render
import pandas as pd


TASK_NAME = "barcode_v2"

FRAGILE_TYPES = {
    "a": "is_fragile_a",
    "b": "is_fragile_b",
    "c": "is_fragile_c",
}


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(TASK_NAME, help="Barcode fragile class plots v2")
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


def _add_flags(df: pd.DataFrame, alexnet_df: pd.DataFrame) -> pd.DataFrame:
    df = get_absolute_fragile(df)
    df = get_relative_drop_fragile(df)
    df = get_rmce_fragile(df, alexnet_df)
    return df


def run(args: argparse.Namespace) -> None:
    out_base = Path(args.output_dir)

    for exp_name, variations in EXPERIMENTS.items():
        print(f"\n[barcode_v2] experiment: {exp_name}")
        alexnet_df = get_rmce_alexnet_df(variations, args.data_path)
        raw_dfs = get_dfs_for_all_models(variations, args.data_path)

        flagged = {
            MODELS[k]: _add_flags(v, alexnet_df)
            for k, v in raw_dfs.items()
            if k in MODELS
        }

        # Individual flag barcodes (A, B, C)
        for type_name, flag_col in FRAGILE_TYPES.items():
            matrix = build_barcode_matrix(flagged, flag_col)
            out = out_base / "images" / "v2" / "barcode" / exp_name / f"{type_name}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            render(matrix, out)
            print(f"  {exp_name}/{type_name}.png")

        # Super-fragile per definition
        for def_name, definition in DEFINITIONS.items():
            super_flagged = {}
            for model_label, df in flagged.items():
                df = df.copy()
                df["is_super_fragile"] = definition.combine(df).astype(int)
                super_flagged[model_label] = df

            matrix = build_barcode_matrix(super_flagged, "is_super_fragile")
            out = (
                out_base
                / "images"
                / "v2"
                / "barcode"
                / exp_name
                / "super_fragile"
                / f"{def_name}.png"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            render(matrix, out)
            print(f"  {exp_name}/super_fragile/{def_name}.png")
