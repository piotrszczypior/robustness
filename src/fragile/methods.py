import pandas as pd
import numpy as np


def calculate_relative_drop(df: pd.DataFrame):
    df = df.copy()

    df["rel_drop"] = ((df["acc_clean"] - df["acc_corrupt"]) / df["acc_clean"]).astype(
        float
    )

    return df


def calculate_absolute_drop(df: pd.DataFrame):
    df = df.copy()
    df["abs_drop"] = (df["acc_clean"] - df["acc_corrupt"]).astype(float)

    return df


def calculate_rmce_mce(
    df_model: pd.DataFrame,
    df_alexnet: pd.DataFrame,
) -> pd.DataFrame:
    merged = df_model.merge(df_alexnet, on="synset", suffixes=("", "_alexnet"))

    numerator_rmce = (1 - merged["acc_corrupt"]) - (1 - merged["acc_clean"])
    denominator_rmce = (1 - merged["acc_corrupt_alexnet"]) - (
        1 - merged["acc_clean_alexnet"]
    )

    numerator_mce = 1 - merged["acc_corrupt"]
    denominator_mce = 1 - merged["acc_corrupt_alexnet"]

    result = merged[["synset"]].copy()
    result["RmCE"] = numerator_rmce / denominator_rmce
    result["mCE"] = numerator_mce / denominator_mce

    df_model = df_model.merge(
        result[["synset", "RmCE", "mCE"]], on="synset", how="left"
    )

    return df_model


def calculate_nCE(
    df_model: pd.DataFrame,
    df_all_models: list[pd.DataFrame],
) -> pd.DataFrame:
    combined = pd.concat(df_all_models)
    baseline = (
        combined.groupby("synset")
        .agg(
            acc_clean_base=("acc_clean", "mean"),
            acc_corrupt_base=("acc_corrupt", "mean"),
        )
        .reset_index()
    )

    merged = df_model.merge(baseline, on="synset")

    numerator = (1 - merged["acc_corrupt"]) - (1 - merged["acc_clean"])
    denominator = (
        (1 - merged["acc_corrupt_base"]) - (1 - merged["acc_clean_base"])
    ).replace(0, np.nan)

    df_model = df_model.copy()
    df_model["nCE"] = (numerator / denominator).values

    return df_model


def calculate_csi(
    df_clean: pd.DataFrame,
    df_sev1: pd.DataFrame,
    df_sev5: pd.DataFrame,
) -> pd.DataFrame:
    """
        CSI - corruption sensitivty index 
    """
    merged = df_clean[["synset", "acc_clean"]].merge(
        df_sev1[["synset", "acc_corrupt"]].rename(columns={"acc_corrupt": "acc_sev1"}),
        on="synset",
    ).merge(
        df_sev5[["synset", "acc_corrupt"]].rename(columns={"acc_corrupt": "acc_sev5"}),
        on="synset",
    )
    merged["CSI"] = (merged["acc_sev1"] - merged["acc_sev5"]) / merged["acc_clean"].replace(0, np.nan)
    return merged[["synset", "CSI"]]
 

def calculate_ccv(
df_clean: pd.DataFrame,
dfs_per_corruption: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
        CCV - cross corruption variance
    """
    acc_cols = {}
    for corruption, df in dfs_per_corruption.items():
        acc_cols[corruption] = df.set_index("synset")["acc_corrupt"]

    acc_matrix = pd.DataFrame(acc_cols)
    result = acc_matrix.std(axis=1).rename("CCV").reset_index()
    return result


def calculate_rrc(dfs_per_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
        RRC - Robustness Rank Consistency
    """
    rank_matrix = {}
    for model, df in dfs_per_model.items():
        ranked = df.set_index("synset")["acc_corrupt"].rank(ascending=False)
        rank_matrix[model] = ranked
 
    rank_df = pd.DataFrame(rank_matrix)
    n_models = len(dfs_per_model)
    n_classes = len(rank_df)
 
    mean_rank = rank_df.mean(axis=1)
    ss_total = ((rank_df.sub(mean_rank, axis=0)) ** 2).sum().sum()
    ss_between = n_models * ((mean_rank - (n_classes + 1) / 2) ** 2).sum()
 
    w = ss_between / (ss_total if ss_total > 0 else np.nan)
 
    rank_std = rank_df.std(axis=1).rename("rank_std")
    result = rank_std.reset_index()
    result["RRC"] = 1 - (result["rank_std"] / rank_df.max().max())
    result.attrs["kendall_w"] = w

    return result[["synset", "rank_std", "RRC"]]


def calculate_crs(dfs_per_corruption: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
        CRS - corruption rank stability
    """
    rank_matrix = {}
    for corruption, df in dfs_per_corruption.items():
        ranked = df.set_index("synset")["acc_corrupt"].rank(ascending=False)
        rank_matrix[corruption] = ranked

    rank_df = pd.DataFrame(rank_matrix)
    rank_std = rank_df.std(axis=1).rename("rank_std")
    result = rank_std.reset_index()
    result["CRS"] = 1 - (result["rank_std"] / rank_df.max().max())
    return result[["synset", "rank_std", "CRS"]]

