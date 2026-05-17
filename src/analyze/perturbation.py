import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

# Typical AlexNet/ResNet errors on clean ImageNet, adjust if using standard mCE denominator
ALEXNET_ERRORS = {
    "defocus_blur": 0.816,
    "glass_blur": 0.826,
    "motion_blur": 0.786,
    "zoom_blur": 0.798,
    "snow": 0.867,
    "frost": 0.827,
    "fog": 0.819,
    "brightness": 0.565,
    "contrast": 0.853,
    "elastic_transform": 0.849,
    "pixelate": 0.732,
    "jpeg_compression": 0.607,
    "gaussian_noise": 0.886,
    "shot_noise": 0.894,
    "impulse_noise": 0.923,
}


def calculate_perturbation_metrics(
    subgroup_dfs: dict[str, pd.DataFrame],
    clean_df: Optional[pd.DataFrame] = None,
    group_name: str = "blur",
) -> pd.DataFrame:
    """
    Calculates average accuracy and mCE per class for a given perturbation group.

    Args:
        subgroup_dfs: A dictionary mapping subgroup names (e.g. 'defocus_blur') to their DataFrames.
                      Each DataFrame must have 'y_true', 'synset', 'is_correct' columns.
        clean_df: Optional. The clean dataset DataFrame with 'y_true', 'synset', 'is_correct'.
        group_name: The name of the perturbation group for logging purposes.

    Returns:
        A DataFrame with the calculated metrics per class:
        ['y_true', 'synset', 'clean_accuracy', 'avg_corrupted_accuracy', 'mCE']
    """
    logger.info(f"Calculating metrics for perturbation group: {group_name}")

    # Calculate clean accuracy if available
    if clean_df is not None:
        clean_acc = (
            clean_df.groupby(["y_true", "synset"])["is_correct"]
            .agg(clean_accuracy="mean")
            .reset_index()
        )
    else:
        # Use first subgroup to extract y_true and synset
        first_df = list(subgroup_dfs.values())[0]
        clean_acc = (
            first_df[["y_true", "synset"]].drop_duplicates().reset_index(drop=True)
        )
        clean_acc["clean_accuracy"] = pd.NA

    # Calculate accuracy per subgroup per class
    subgroup_accs = {}
    for sub_name, df in subgroup_dfs.items():
        acc = (
            df.groupby(["y_true", "synset"])["is_correct"]
            .agg(accuracy="mean")
            .reset_index()
        )
        subgroup_accs[sub_name] = acc

    # Merge all accuracies into one dataframe
    merged_df = clean_acc.copy()

    for sub_name, acc_df in subgroup_accs.items():
        # Rename column to include subgroup name
        acc_df = acc_df.rename(columns={"accuracy": f"acc_{sub_name}"})
        merged_df = pd.merge(merged_df, acc_df, on=["y_true", "synset"], how="left")

    # Calculate average corrupted accuracy across the subgroups
    acc_columns = [f"acc_{sub_name}" for sub_name in subgroup_dfs.keys()]
    merged_df["avg_corrupted_accuracy"] = merged_df[acc_columns].mean(axis=1)

    # Calculate CE for each subgroup, then mCE
    # CE_c = Error_c / Baseline_Error_c (for AlexNet typically, or just clean_error)
    # If using AlexNet baselines:
    ce_columns = []
    for sub_name in subgroup_dfs.keys():
        acc_col = f"acc_{sub_name}"
        error_col = f"err_{sub_name}"
        ce_col = f"ce_{sub_name}"

        merged_df[error_col] = 1.0 - merged_df[acc_col]

        baseline_err = ALEXNET_ERRORS.get(
            sub_name, 1.0
        )  # default to 1.0 if not found to just return error
        merged_df[ce_col] = merged_df[error_col] / baseline_err
        ce_columns.append(ce_col)

    merged_df["mCE"] = merged_df[ce_columns].mean(axis=1)

    # Clean up intermediate columns to keep it tidy
    cols_to_drop = (
        acc_columns + [f"err_{sub}" for sub in subgroup_dfs.keys()] + ce_columns
    )
    final_df = merged_df.drop(columns=cols_to_drop)

    return final_df
