import pandas as pd
import os
from space import CorruptionVariations


def get_files(model_prefix: str, results_path: str = "results/") -> list[str]:
    """Return the full list of expected CSV paths for *model_prefix*.

    Includes one clean baseline file and one file per (corruption, severity)
    combination defined by :class:`VariationSpaceImageNetC`.
    """
    all_files: list[str] = []

    for var in CorruptionVariations(models=[model_prefix]):
        filename = (
            f"{model_prefix}_imagenet_c_{var.group}_{var.corruption}_{var.severity}.csv"
        )
        all_files.append(os.path.join(results_path, filename))

    all_files.append(os.path.join(results_path, f"{model_prefix}_imagenet.csv"))

    return all_files


def load_and_aggregate_results(model_prefix, results_path="results/"):
    files = get_files(model_prefix, results_path)

    existing = [f for f in files if os.path.exists(f)]

    if not existing:
        raise FileNotFoundError(f"Files missing. Model: {model_prefix}")

    aggregated: list[pd.DataFrame] = []
    for filepath in existing:
        df = pd.read_csv(filepath)

        per_class = df.groupby("synset")["is_correct"].mean().reset_index()
        per_class = per_class.rename(columns={"is_correct": "accuracy"})

        raw_corr = df["corruption"].iloc[0]
        if pd.isna(raw_corr) or str(raw_corr).lower() == "none":
            corr_type = "clean"
        else:
            corr_type = raw_corr

        raw_sev = df["severity"].iloc[0]
        sev_level = 0 if corr_type == "clean" else raw_sev

        per_class["corruption"] = corr_type
        per_class["severity"] = sev_level

        aggregated.append(per_class[["synset", "corruption", "severity", "accuracy"]])

    return pd.concat(aggregated, ignore_index=True)


def aggregate_for_rmce(
    df: pd.DataFrame,
    aggragated_column_name: str = "corrupted",
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
) -> pd.DataFrame:
    clean = (
        df[df["corruption"] == "clean"]
        .groupby("synset")["accuracy"]
        .mean()
        .rename("clean")
    )

    mask = df["corruption"] != "clean"
    if corruptions is not None:
        mask &= df["corruption"].isin(corruptions)
    if severities is not None:
        mask &= df["severity"].isin(severities)

    corrupted = (
        df[mask].groupby("synset")["accuracy"].mean().rename(aggragated_column_name)
    )

    return pd.concat([clean, corrupted], axis=1).reset_index()


def compute_rmce_mce(
    df_model: pd.DataFrame,
    df_alexnet: pd.DataFrame,
    corruption_group_name: str = "corrupted",
) -> pd.DataFrame:
    merged = df_model.merge(df_alexnet, on="synset", suffixes=("", "_alex"))

    numerator_rmce = (1 - merged[corruption_group_name]) - (1 - merged["clean"])
    denominator_rmce = (1 - merged[f"{corruption_group_name}_alex"]) - (
        1 - merged["clean_alex"]
    )

    numerator_mce = 1 - merged[corruption_group_name]
    denominator_mce = 1 - merged[f"{corruption_group_name}_alex"]

    result = merged[["synset"]].copy()
    result["RmCE"] = numerator_rmce / denominator_rmce
    result["mCE"] = numerator_mce / denominator_mce

    return result


def get_denom_indices(alexnet_df: pd.DataFrame, donom_min=0.05):
    denom = alexnet_df.set_index("synset").eval("clean - corrupted")
    stable_synsets = denom[denom.abs() > donom_min].index
    return stable_synsets


# def compute_mce(
#     df_model: pd.DataFrame,
#     df_alexnet: pd.DataFrame,
#     corruption_group_name: str,
# ) -> pd.DataFrame:
#     merged = df_model.merge(df_alexnet, on="synset", suffixes=("", "_alex"))

#     numerator   = 1 - merged[corruption_group_name]
#     denominator = 1 - merged[f"{corruption_group_name}_alex"]

#     result = merged[["synset"]].copy()
#     result["mCE"] = numerator / denominator
#     return result


if __name__ == "__main__":
    res_model = load_and_aggregate_results("resnet50")
    res_alexnet = load_and_aggregate_results("alexnet")

    # agg = aggregate_for_rmce(
    #     df=res,
    #     corruptions=["defocus_blur", "zoom_blur"],
    #     severities=[1, 2, 3],
    #     corruption_group_name="blur",
    # )

    # agg_model   = aggregate_for_rmce(res_model,   ["defocus_blur", "zoom_blur"], [1,2,3], "blur")
    # agg_alexnet = aggregate_for_rmce(res_alexnet, ["defocus_blur", "zoom_blur"], [1,2,3], "blur")

    # rmce = compute_rmce(agg_model, agg_alexnet, "blur")

    # print(rmce.head(10))

    # print(rmce["RmCE"].mean())

    agg_model2 = aggregate_for_rmce(res_model, "all")
    agg_alexnet2 = aggregate_for_rmce(res_alexnet, "all")

    rmce = compute_rmce_mce(agg_model2, agg_alexnet2, "all")
    mce = compute_rmce_mce(agg_model2, agg_alexnet2, "all")

    # print(rmce["RmCE"].mean())
    # print(mce["mCE"].mean())

    sort = rmce.sort_values(by=["mCE"])

    print(sort.head(10))

    print(sort.head(10).to_latex())
