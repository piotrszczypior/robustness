import json
import logging
from pathlib import Path
from typing import Protocol

import pandas as pd

logger = logging.getLogger(__name__)


class FragileStrategy(Protocol):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a fragility filter to a DataFrame.
        Expected columns: 'accuracy_clean', 'accuracy_corrupted', 'y_true', 'synset'.
        Returns a DataFrame containing only the fragile classes.
        """
        ...


class ThresholdFilter:
    def __init__(self, clean_min: float = 0.8, corrupt_max: float = 0.5):
        self.clean_min = clean_min
        self.corrupt_max = corrupt_max

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            (df["accuracy_clean"] >= self.clean_min)
            & (df["accuracy_corrupted"] <= self.corrupt_max)
        ]


class TailFilter:
    def __init__(self, k: int = 25, sort_by: str = "worst"):
        """
        k: number of classes to select
        sort_by: "worst" (lowest corrupted accuracy) or "best" (highest corrupted accuracy)
        """
        self.k = k
        self.sort_by = sort_by
        if self.sort_by not in ("worst", "best"):
            raise ValueError("sort_by must be 'worst' or 'best'")

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        ascending = True if self.sort_by == "worst" else False
        return df.sort_values(by="accuracy_corrupted", ascending=ascending).head(self.k)


class AccuracyDropFilter:
    def __init__(self, k: int = 15):
        """
        Selects top k classes with the largest difference between clean and corrupted accuracy.
        """
        self.k = k

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["accuracy_drop"] = df["accuracy_clean"] - df["accuracy_corrupted"]
        return df.sort_values(by="accuracy_drop", ascending=False).head(self.k)


def merge_and_calculate_accuracies(
    baseline_df: pd.DataFrame, corrupted_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates accuracy per class for baseline and corrupted dataframes and merges them.
    Expects columns 'y_true', 'synset', 'is_correct' in the input dataframes.
    """
    baseline_acc = (
        baseline_df.groupby(["y_true", "synset"])["is_correct"]
        .agg(accuracy="mean")
        .reset_index()
    )
    corrupted_acc = (
        corrupted_df.groupby(["y_true", "synset"])["is_correct"]
        .agg(accuracy="mean")
        .reset_index()
    )

    merged = pd.merge(baseline_acc, corrupted_acc, on=["y_true", "synset"]).rename(
        columns={"accuracy_x": "accuracy_clean", "accuracy_y": "accuracy_corrupted"}
    )
    return merged


def find_overlapping_fragile_classes(
    domain_pairs: list[tuple[pd.DataFrame, pd.DataFrame]],
    strategy: FragileStrategy,
) -> pd.DataFrame:
    """
    domain_pairs: A list of (clean_df, corrupted_df) tuples for different domains (corruptions).
    strategy: The filter strategy to apply.
    Returns the overlapping fragile classes across all provided domains.
    """
    if not domain_pairs:
        return pd.DataFrame()

    fragile_sets = []
    merged_dfs = []
    for baseline_df, corrupted_df in domain_pairs:
        merged = merge_and_calculate_accuracies(baseline_df, corrupted_df)
        fragile = strategy(merged)
        fragile_sets.append(set(fragile["y_true"].tolist()))
        merged_dfs.append(merged)

    # Intersection of all fragile classes
    common_y_trues = set.intersection(*fragile_sets) if fragile_sets else set()

    # Create a summary dataframe for the overlapping classes based on the first domain
    # Or average accuracies across domains (here we just use the first domain's clean accuracy as reference)
    base_df = merged_dfs[0][merged_dfs[0]["y_true"].isin(common_y_trues)].copy()

    # Average the corrupted accuracy across all domains for these common fragile classes
    avg_corrupted = (
        pd.concat(merged_dfs)
        .groupby("y_true")["accuracy_corrupted"]
        .mean()
        .reset_index()
    )
    avg_corrupted.rename(
        columns={"accuracy_corrupted": "avg_accuracy_corrupted"}, inplace=True
    )

    result = pd.merge(
        base_df[["y_true", "synset", "accuracy_clean"]], avg_corrupted, on="y_true"
    )
    return result


def export_to_json(df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path_with_file = output_dir / filename
    output = {"classes": df.to_dict(orient="records")}
    with open(path_with_file, "w") as f:
        json.dump(output, f, indent=4)
    logger.info(f"Saved JSON to {path_with_file}")


def export_to_latex(
    df: pd.DataFrame,
    caption: str = "Fragile Classes",
    label: str = "tab:fragile_classes",
) -> str:
    """
    Exports the DataFrame to a formatted LaTeX table string.
    """
    # Select subset of columns to display if available
    display_cols = ["y_true", "synset", "accuracy_clean"]
    if "avg_accuracy_corrupted" in df.columns:
        display_cols.append("avg_accuracy_corrupted")
    elif "accuracy_corrupted" in df.columns:
        display_cols.append("accuracy_corrupted")
    if "accuracy_drop" in df.columns:
        display_cols.append("accuracy_drop")

    cols_to_use = [c for c in display_cols if c in df.columns]

    latex_str = df.to_latex(
        columns=cols_to_use,
        index=False,
        caption=caption,
        label=label,
        float_format="%.4f",
    )
    return latex_str
