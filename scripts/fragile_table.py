from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from fragile.experiments import EXPERIMENTS, get_dfs_for_all_models, get_rmce_alexnet_df
from fragile.fragile import get_absolute_fragile, get_relative_drop_fragile, get_rmce_fragile
from fragile.definitions import DEFINITIONS
from plots_v2.barcode.plot import build_barcode_matrix
from utils import get_index_to_synset_and_label_imagenet1k
from model import MODELS


def _add_flags(df: pd.DataFrame, alexnet_df: pd.DataFrame) -> pd.DataFrame:
    df = get_absolute_fragile(df)
    df = get_relative_drop_fragile(df)
    df = get_rmce_fragile(df, alexnet_df)
    return df


def build_table(variations, data_path: str) -> pd.DataFrame:
    alexnet_df = get_rmce_alexnet_df(variations, data_path)
    raw_dfs = get_dfs_for_all_models(variations, data_path)

    ab = DEFINITIONS["ab"]
    super_flagged = {}
    for k, df in raw_dfs.items():
        if k not in MODELS:
            continue
        df = _add_flags(df.copy(), alexnet_df)
        df["is_super_fragile"] = ab.combine(df).astype(int)
        super_flagged[MODELS[k]] = df

    matrix = build_barcode_matrix(super_flagged, "is_super_fragile")

    common_indices = sorted(
        col for col in matrix.columns if (matrix[col] == 2).any()
    )

    index_to_info = get_index_to_synset_and_label_imagenet1k()
    rows = [
        {
            "index": idx,
            "synset": index_to_info[idx][0],
            "label": index_to_info[idx][1],
        }
        for idx in common_indices
        if idx in index_to_info
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Table of common fragile classes")
    parser.add_argument("--data-path", default="results")
    parser.add_argument(
        "--exp",
        default="all_corruptions",
        choices=list(EXPERIMENTS.keys()),
    )
    parser.add_argument("--output", default="results/fragile_table.csv")
    args = parser.parse_args()

    variations = EXPERIMENTS[args.exp]
    table = build_table(variations, args.data_path)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    print(table.to_string(index=False))
    print(f"\n{len(table)} common fragile classes → {out}")


if __name__ == "__main__":
    main()
