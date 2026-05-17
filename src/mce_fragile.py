"""
Identify common fragile classes across architectures.
Outputs a CSV table: synset, class_name, fragile_count, fragile_models, mean_RmCE
"""

from __future__ import annotations
import argparse
import pandas as pd

from mce import (
    load_and_aggregate_results,
    aggregate_for_rmce,
    compute_rmce_mce,
    get_denom_indices,
)
from space import CorruptionVariations
from model import MODELS
from utils import get_synset_to_label_imagenet1k

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = "results"
RMCE_THRESHOLD = 2.0
MIN_MODELS = 18


def build_fragile_table(
    data_dir: str,
    group_name: str,
    corruptions_filter: list[str] | None,
    severities_filter: list[int] | None,
    rmce_threshold: float,
    min_models: int,
) -> pd.DataFrame:
    models = [m for m in MODELS.keys() if m != "alexnet"]

    df_alexnet = load_and_aggregate_results("alexnet", data_dir)

    vs = CorruptionVariations(
        groups=[group_name],
        corruptions=corruptions_filter,
        severities=severities_filter,
    )
    group_corruptions = list(set(v.corruption for v in vs))

    agg_alex = aggregate_for_rmce(
        df_alexnet, corruptions=group_corruptions, severities=severities_filter
    )
    stable_synsets = set(get_denom_indices(agg_alex))

    # collect per-model RmCE
    all_rmce: dict[str, pd.Series] = {}

    for model_name in models:
        df_model = load_and_aggregate_results(model_name, data_dir)
        agg_model = aggregate_for_rmce(
            df_model, corruptions=group_corruptions, severities=severities_filter
        )
        rmce_df = compute_rmce_mce(agg_model, agg_alex)
        rmce_df = rmce_df[rmce_df["synset"].isin(stable_synsets)]
        all_rmce[MODELS[model_name]] = rmce_df.set_index("synset")["RmCE"]

    rmce_wide = pd.DataFrame(all_rmce)  # synset × model

    # fragile flag per cell
    fragile = (rmce_wide > rmce_threshold).astype(int)

    fragile_count = fragile.sum(axis=1)
    fragile_models = fragile.apply(
        lambda row: ", ".join(row.index[row == 1].tolist()), axis=1
    )
    mean_rmce = rmce_wide.mean(axis=1)
    max_rmce = rmce_wide.max(axis=1)

    result = pd.DataFrame(
        {
            "synset": fragile_count.index,
            "fragile_count": fragile_count.values,
            "mean_RmCE": mean_rmce.values,
            "max_RmCE": max_rmce.values,
            "most_fragile_model": rmce_wide.idxmax(axis=1).values,
        }
    )

    result = result[result["fragile_count"] >= min_models]
    result = result.sort_values("fragile_count", ascending=False)

    synset_to_name = get_synset_to_label_imagenet1k()
    result.insert(1, "class_name", result["synset"].map(synset_to_name).fillna(""))

    return result.reset_index(drop=True)


def main(args):
    print(
        f"Building fragile class table — group={args.group}, "
        f"RmCE>{args.rmce_threshold}, min_models={args.min_models}"
    )

    table = build_fragile_table(
        data_dir=args.data_dir,
        group_name=args.group,
        corruptions_filter=args.corruptions,
        severities_filter=args.severities,
        rmce_threshold=args.rmce_threshold,
        min_models=args.min_models,
    )

    print(f"\nFound {len(table)} fragile classes:\n")
    print(table.to_string(index=False))

    print()

    print(table.to_latex())

    # out = Path(args.output)
    # out.parent.mkdir(parents=True, exist_ok=True)
    # table.to_csv(out, index=False)
    # print(f"\nSaved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--group", default="blur")
    parser.add_argument("--corruptions", nargs="*", default=None)
    parser.add_argument("--severities", nargs="*", type=int, default=None)
    parser.add_argument("--rmce-threshold", type=float, default=RMCE_THRESHOLD)
    parser.add_argument("--min-models", type=int, default=MIN_MODELS)
    parser.add_argument("--output", default="fragile_classes.csv")
    args = parser.parse_args()
    main(args)
