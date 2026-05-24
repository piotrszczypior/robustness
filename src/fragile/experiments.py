from model import MODELS
from space import CorruptionVariations
from .data import get_per_class_accuracy
from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES
from pathlib import Path
import pandas as pd
from .methods import (
    calculate_relative_drop,
    calculate_absolute_drop,
    calculate_rmce_mce,
    calculate_nCE,
)


_STANDARD_GROUPS = [k for k in IMAGENET_C_CORRUPTION_GROUPS if k != "extra"]

EXPERIMENTS = {
    "all_corruptions": CorruptionVariations(
        groups=_STANDARD_GROUPS,
        severities=IMAGENET_C_SEVERITIES,
    ),
    "severity_1": CorruptionVariations(
        groups=_STANDARD_GROUPS,
        severities=[1],
    ),
    "blur": CorruptionVariations(
        groups=["blur"],
        severities=IMAGENET_C_SEVERITIES,
    ),
    "noise": CorruptionVariations(
        groups=["noise"],
        severities=IMAGENET_C_SEVERITIES,
    ),
    "weather": CorruptionVariations(
        groups=["weather"],
        severities=IMAGENET_C_SEVERITIES,
    ),
    "digital": CorruptionVariations(
        groups=["digital"],
        severities=IMAGENET_C_SEVERITIES,
    ),
    "blur_1": CorruptionVariations(
        groups=["blur"],
        severities=[1],
    ),
    "noise_1": CorruptionVariations(
        groups=["noise"],
        severities=[1],
    ),
    "weather_1": CorruptionVariations(
        groups=["weather"],
        severities=[1],
    ),
    "digital_1": CorruptionVariations(
        groups=["digital"],
        severities=[1],
    ),
}


def _get_clean_per_class(model: str, data_dir: Path) -> pd.DataFrame:
    return get_per_class_accuracy(
        f"{model}_imagenet.csv", data_dir, agg_column="acc_clean"
    )


def _get_per_class_by_alias(model: str, alias: str, data_path: Path) -> pd.DataFrame:
    return get_per_class_accuracy(
        f"{model}_{alias}.csv", data_path, agg_column=f"acc_corrupt"
    )


def _get_corrupt_per_class(
    model: str, variations: CorruptionVariations, data_dir: Path
) -> pd.DataFrame:
    frames = []
    for group, corruption, severity in variations.per_unique_conditions():
        fname = f"{model}_imagenet_c_{group}_{corruption}_{severity}.csv"

        try:
            df = get_per_class_accuracy(fname, data_dir)[["synset", "accuracy"]]
            frames.append(df)
        except FileNotFoundError:
            continue

    if not frames:
        return pd.DataFrame(columns=["synset", "acc_corrupt"])

    return (
        pd.concat(frames)
        .groupby("synset")["accuracy"]
        .mean()
        .rename("acc_corrupt")
        .reset_index()
    )


def _build_single_df(
    model: str,
    variations: CorruptionVariations,
    alexnet_df: pd.DataFrame,
    data_path="results",
):
    clean = _get_clean_per_class(model, data_path)
    corrupt = _get_corrupt_per_class(model, variations, data_path)
    df = clean.merge(corrupt, on="synset").dropna()
    df = calculate_relative_drop(df)
    df = calculate_absolute_drop(df)
    df = calculate_rmce_mce(df, alexnet_df)

    return df


def _build_single_df_by_alias(
    model: str,
    alias: str,
    data_path="results",
):
    clean = _get_clean_per_class(model, data_path)
    corrupt = _get_per_class_by_alias(model, alias, data_path)
    df = clean.merge(corrupt, on=["synset", "y_true"], how="right").dropna()
    df = calculate_relative_drop(df)
    df = calculate_absolute_drop(df)
    return df


def get_rmce_alexnet_df(variations, data_path="results"):
    alexnet_clean = _get_clean_per_class("alexnet", data_path)
    alexnet_corrupt = _get_corrupt_per_class("alexnet", variations, data_path)
    return alexnet_clean.merge(alexnet_corrupt, on="synset").dropna()


def get_dfs_for_all_models(variations: CorruptionVariations, data_path="results"):
    alexnet_rmce_df = get_rmce_alexnet_df(variations, data_path)

    dfs = {}
    for model in MODELS.keys():
        dfs[model] = _build_single_df(model, variations, alexnet_rmce_df, data_path)

    all_dfs = list(dfs.values())
    for model in dfs:
        dfs[model] = calculate_nCE(dfs[model], all_dfs)

    return dfs


def get_df_for_model(variations: CorruptionVariations, model: str, data_path="results"):
    alexnet_rmce_df = get_rmce_alexnet_df(variations, data_path)
    return _build_single_df(model, variations, alexnet_rmce_df, data_path)


def get_dfs_for_experiment(experiment: str, model: str, data_path="results"):
    variation = EXPERIMENTS[experiment]
    alexnet_rmce_df = get_rmce_alexnet_df(variation, data_path)

    return _build_single_df(model, variation, alexnet_rmce_df, data_path)


def get_dfs_for_dataset(
    dataset_alias: str, data_path: str = "results"
) -> dict[str, pd.DataFrame]:
    dfs = {}

    for model in MODELS.keys():
        try:
            dfs[model] = _build_single_df_by_alias(model, dataset_alias, data_path)
        except FileNotFoundError:
            continue

    return dfs


def get_alexnet_df_by_alias(alias: str, data_path: str = "results") -> pd.DataFrame:
    return _build_single_df_by_alias("alexnet", alias, data_path)


def get_df_by_alias_with_rmce(
    model: str, alias: str, alexnet_df: pd.DataFrame, data_path: str = "results"
) -> pd.DataFrame:
    df = _build_single_df_by_alias(model, alias, data_path)
    return calculate_rmce_mce(df, alexnet_df)
