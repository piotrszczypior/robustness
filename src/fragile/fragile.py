import pandas as pd
import numpy as np

from .definitions import FragileDefinition


BASELINE_ACCURACY = 0.80
CUTOFF_ACCURACY = 0.50
RMCE_CUTOFF = 1.5
RELATIVE_DROP_PERCENTILE = 75


def get_absolute_fragile(
    df: pd.DataFrame,
    baseline: float = BASELINE_ACCURACY,
    cutoff: float = CUTOFF_ACCURACY,
) -> pd.DataFrame:
    df = df.copy()
    df["is_fragile_a"] = (
        (df["acc_clean"] >= baseline) & (df["acc_corrupt"] <= cutoff)
    ).astype(int)

    return df


def get_relative_drop_fragile(
    df: pd.DataFrame, percentile: float = RELATIVE_DROP_PERCENTILE
) -> pd.DataFrame:
    df = df.copy()
    threshold = np.percentile(df["rel_drop"].dropna(), percentile)
    df["is_fragile_b"] = (df["rel_drop"] >= threshold).astype(int)
    df.attrs["threshold"] = threshold

    return df


def get_rmce_fragile(
    df: pd.DataFrame,
    df_alexnet: pd.DataFrame,
    rmce_cutoff: float = RMCE_CUTOFF,
    denom_min: float = 0.05,
) -> pd.DataFrame:
    df = df.copy()
    stable_synsets = _get_denom_indices(df_alexnet, denom_min=denom_min)

    df["is_fragile_c"] = (
        df["synset"].isin(stable_synsets) & (df["RmCE"] >= rmce_cutoff)
    ).astype(int)

    return df


def get_fragile_by_definition(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    # df_c: pd.DataFrame,
    definition: FragileDefinition,
) -> pd.DataFrame:
    merged = df_a[["synset", "is_fragile_a"]]
    merged = merged.merge(df_b[["synset", "is_fragile_b"]], on="synset")
    # merged = merged.merge(df_c[["synset", "is_fragile_c"]], on="synset")
    merged["is_strongly_fragile"] = definition.combine(merged).astype(int)
    return merged


def get_cross_model_fragile(
    dfs: list[pd.DataFrame],
    definition: FragileDefinition,
    min_models: int = 16,
) -> pd.DataFrame:
    fragile_dfs = [df[definition.combine(df)] for df in dfs]

    combined = pd.concat(fragile_dfs)

    agg = (
        combined.groupby("synset")
        .agg(
            y_true=("y_true", "first"),
            acc_clean=("acc_clean", "mean"),
            acc_corrupt=("acc_corrupt", "mean"),
            rel_drop=("rel_drop", "mean"),
            abs_drop=("abs_drop", "mean"),
            mCE=("mCE", "mean"),
            RmCE=("RmCE", "mean"),
            nCE=("nCE", "mean"),
            fragile_count=("acc_clean", "count"),
        )
        .reset_index()
    )

    return agg[agg["fragile_count"] >= min_models].sort_values(
        "fragile_count", ascending=False
    )


def get_cross_model_df(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(dfs)

    agg = (
        combined.groupby("synset")
        .agg(
            y_true=("y_true", "first"),
            acc_clean=("acc_clean", "mean"),
            acc_corrupt=("acc_corrupt", "mean"),
            rel_drop=("rel_drop", "mean"),
            abs_drop=("abs_drop", "mean"),
            mCE=("mCE", "mean"),
            RmCE=("RmCE", "mean"),
            nCE=("nCE", "mean"),
            fragile_count=("is_strongly_fragile", "sum"),
        )
        .reset_index()
    )

    return agg


def get_cross_model_df(
    dfs: list[pd.DataFrame],
    agg_cols: list[str] = ["acc_clean", "acc_corrupt", "rel_drop", "abs_drop"],
) -> pd.DataFrame:
    combined = pd.concat(dfs)
    agg_dict = {col: (col, "mean") for col in agg_cols if col in combined.columns}
    agg_dict["y_true"] = ("y_true", "first")

    agg = combined.groupby("synset").agg(**agg_dict).reset_index()
    return agg


def _pareto_front(df: pd.DataFrame) -> pd.Index:
    clean = df["acc_clean"].values
    corrupt = df["acc_corrupt"].values
    n = len(df)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if clean[j] >= clean[i] and corrupt[j] <= corrupt[i]:
                if clean[j] > clean[i] or corrupt[j] < corrupt[i]:
                    is_dominated[i] = True
                    break
    return df.index[~is_dominated]


def select_top_k_fragile(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df.sort_values(by="acc_clean", ascending=True)[:k]

    if "is_strongly_fragile" in df.columns:
        candidates = df[df["is_strongly_fragile"] == 1].copy()
    else:
        candidates = df.copy()

    selected = []
    remaining = candidates.copy()

    while len(selected) < k and len(remaining) > 0:
        front_idx = _pareto_front(remaining)
        front = remaining.loc[front_idx]
        needed = k - len(selected)

        if len(front) <= needed:
            selected.append(front)
        else:
            tiebreak_col = "abs_drop" if "abs_drop" in front.columns else "acc_clean"
            selected.append(front.nlargest(needed, tiebreak_col))
            break

        remaining = remaining.drop(front_idx)

    return pd.concat(selected) if selected else candidates.iloc[0:0]


def _pareto_front_robust(df: pd.DataFrame) -> pd.Index:
    clean = df["acc_clean"].values
    corrupt = df["acc_corrupt"].values
    n = len(df)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if clean[j] >= clean[i] and corrupt[j] >= corrupt[i]:
                if clean[j] > clean[i] or corrupt[j] > corrupt[i]:
                    is_dominated[i] = True
                    break
    return df.index[~is_dominated]


def select_top_k_robust(df: pd.DataFrame, k: int) -> pd.DataFrame:
    candidates = df.copy()
    selected = []
    remaining = candidates.copy()

    while len(selected) < k and len(remaining) > 0:
        front_idx = _pareto_front_robust(remaining)
        front = remaining.loc[front_idx]
        needed = k - len(selected)

        if len(front) <= needed:
            selected.append(front)
        else:
            selected.append(front.nlargest(needed, "acc_corrupt"))
            break

        remaining = remaining.drop(front_idx)

    return pd.concat(selected) if selected else candidates.iloc[0:0]


def _get_denom_indices(alexnet_df: pd.DataFrame, denom_min=0.05):
    denom = alexnet_df.set_index("synset").eval("acc_clean - acc_corrupt")
    stable_synsets = denom[denom.abs() > denom_min].index
    return stable_synsets
