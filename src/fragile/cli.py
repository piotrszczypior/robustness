from pathlib import Path

from networkx import to_latex
from task import Task
import argparse
from .experiments import EXPERIMENTS, get_dfs_for_all_models, get_df_for_model, get_dfs_for_experiment, get_rmce_alexnet_df, get_alexnet_df_by_alias, get_df_by_alias_with_rmce
from model import MODELS
from .fragile import (
    get_absolute_fragile,
    get_relative_drop_fragile,
    get_rmce_fragile,
    get_strongly_fragile,
    get_cross_model_fragile,
    get_cross_model_df,
    select_top_k_fragile,
    select_top_k_robust,
)
from .methods import calculate_relative_drop
from .clustering import run_clustering, run_kmeans, run_pca, run_umap, plot_kmeans
from .definitions import DEFINITIONS, FragileDefinition
import pandas as pd
from collections import Counter
import numpy as np
from .representation import cluster_stats_to_latex

TASK_NAME = "fragile"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(TASK_NAME, help="Analyze fragile classes")
    parser.add_argument(
        "--exp",
        type=str,
        default="all_corruptions",
        help="Experiment",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="results",
        help="path to data",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--definition",
        type=str,
        default="ab",
        choices=list(DEFINITIONS.keys()) + ["all"],
        help="Fragile class combination definition (use 'all' to run each definition in sequence)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run all experiments × definitions and save results to disk",
    )
    parser.add_argument(
        "--common",
        action="store_true",
        help="Find synsets fragile across ALL experiments for a given definition",
    )
    parser.add_argument(
        "--clustering-fragile",
        action="store_true",
        help="Run unsupervised clustering algoritm to determine fragile classese",
    ),
    parser.add_argument(
        "--rmce-fragile",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Base output directory for sweep results",
    )
    parser.add_argument(
        "--save-tables",
        action="store_true",
        default=False,
        help="Saves tables to txt file",
    )
    parser.add_argument(
        "--select-top-k",
        type=int,
        default=None,
        help="Select top K fragile classes via Pareto front (max acc_clean, min acc_corrupt)",
    )
    parser.add_argument(
        "--granular-sweep",
        action="store_true",
        help="Count fragile classes per corruption x severity for a single model",
    )
    parser.add_argument(
        "--exp-sweep",
        action="store_true",
        help="Count cross-model fragile classes per experiment, broken down by condition (A, B, definitions)",
    )
    parser.add_argument(
        "--granular-group-sweep",
        action="store_true",
        help="For each corruption group: intersect fragile synsets across all corruptions x severities, filter by Pareto, save LaTeX table",
    )
    parser.add_argument(
        "--granular-group-cross-model",
        action="store_true",
        help="Like --granular-group-sweep but across all models; outputs a cross-model checkmark table",
    )
    parser.add_argument(
        "--granular-group-cross-model-avg",
        action="store_true",
        help="Cross-model checkmark table using per-group averaged metrics (uses get_dfs_for_experiment)",
    )
    parser.add_argument(
        "--intersect-models",
        action="store_true",
        help="Keep only synsets fragile in ALL models for each group (intersection across models)",
    )
    parser.add_argument(
        "--robust-classes",
        action="store_true",
        help="Find top-K robust classes for a single model via Pareto front (acc_clean ↑, acc_corrupt ↑)",
    )
    parser.add_argument(
        "--corruption",
        type=str,
        default=None,
        help="Filter to a single corruption (e.g. defocus_blur); used by --granular-group-cross-model",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=None,
        help="Filter to a single severity level (1–5); used by --granular-group-cross-model",
    )
    parser.add_argument(
        "--dataset-intersection",
        action="store_true",
        help="For a single dataset alias: find synsets fragile in ALL 4 selected models",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset alias for --dataset-intersection / --arch-contrast (e.g. imagenet_c_motion_blur_2)",
    )
    parser.add_argument(
        "--arch-contrast",
        action="store_true",
        help="Find synsets fragile only in ViT models (not CNN) and vice-versa",
    )
    parser.add_argument(
        "--vit-models",
        nargs="+",
        default=["vit_b_16"],
        help="ViT model keys for --arch-contrast",
    )
    parser.add_argument(
        "--cnn-models",
        nargs="+",
        default=["efficientnet_b4"],
        help="CNN model keys for --arch-contrast",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Corruption group for --arch-contrast (e.g. blur); averages across all corruptions x severities in group",
    )
    parser.add_argument(
        "--min-models",
        type=int,
        default=None,
        help="Minimum models per family that must agree (default: all models in the family)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Minimum difference rel_drop_vit - rel_drop_cnn (or vice versa) to qualify as exclusive (default: 0.1)",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Save scatter plot (rel_drop_cnn vs rel_drop_vit) for --arch-contrast",
    )
    parser.add_argument(
        "--arch-contrast-v2",
        action="store_true",
        help="Gap-based architecture contrast (delta_g method, see arch_contrast.py)",
    )
    parser.add_argument(
        "--theta-a",
        type=float,
        default=0.3,
        help="Minimum asymmetry score for --arch-contrast-v2 (default: 0.3)",
    )
    parser.add_argument(
        "--theta-min",
        type=float,
        default=0.1,
        help="Minimum absolute relative drop for --arch-contrast-v2 (default: 0.1)",
    )
    parser.add_argument(
        "--no-pareto",
        action="store_true",
        help="Skip secondary Pareto filter in --arch-contrast-v2",
    )
    parser.add_argument(
        "--fisher-matrix",
        action="store_true",
        help="Compute Fisher exact test p-value matrix (20x20) for fragile class correlation between all model pairs",
    )


def run(args: argparse.Namespace):
    if args.sweep:
        run_sweep(args)
        return

    if args.common:
        run_common(args)
        return

    if args.clustering_fragile:
        run_clustering_fragile_sweep(args)
        return

    if args.rmce_fragile:
        get_fragile_by_rmce(args)
        return

    if args.granular_sweep:
        run_granular_fragile_sweep(args)
        return

    if args.granular_group_sweep:
        run_granular_group_intersection_sweep(args)
        return

    if args.granular_group_cross_model:
        run_granular_group_cross_model_sweep(args)
        return

    if args.granular_group_cross_model_avg:
        run_granular_group_cross_model_avg_sweep(args)
        return

    if args.robust_classes:
        run_robust_classes(args)
        return

    if args.exp_sweep:
        run_experiment_fragile_sweep(args)
        return

    if args.dataset_intersection:
        run_dataset_model_intersection(args)
        return

    if args.arch_contrast:
        run_arch_contrast(args)
        return

    if args.arch_contrast_v2:
        run_arch_contrast_v2(args)
        return

    if args.fisher_matrix:
        run_fisher_matrix(args)
        return

    variations = EXPERIMENTS[args.exp]
    dfs = get_dfs_for_all_models(variations, args.data_path)

    definitions = (
        list(DEFINITIONS.values())
        if args.definition == "all"
        else [DEFINITIONS[args.definition]]
    )

    if args.model:
        for definition in definitions:
            print(f"\n=== {definition.label} ===")
            df = _get_fragile(dfs[args.model], dfs["alexnet"], definition)
            # print(df[df["is_strongly_fragile"] == 1])
            robust = df[
                (df["is_fragile_a"] == 0)
                & (df["is_fragile_b"] == 0)
                & (df["is_fragile_c"] == 0)
            ].sort_values(by="acc_corrupt", ascending=False)
            print(robust)
            if args.select_top_k:
                top_k = select_top_k_fragile(df, args.select_top_k)
                print(f"\n--- Top {args.select_top_k} (Pareto) ---")
                print(top_k[["synset", "acc_clean", "acc_corrupt", "abs_drop"]])
            # print(df.describe())
        return

    for definition in definitions:
        print(f"\n=== {definition.label} ===")
        fragile_dfs = [
            _get_fragile(df, dfs["alexnet"], definition) for df in dfs.values()
        ]
        cross = get_cross_model_fragile(fragile_dfs, definition, min_models=15)
        print(cross)
        print(len(cross))
        if args.select_top_k:
            top_k = select_top_k_fragile(cross, args.select_top_k)
            print(f"\n--- Top {args.select_top_k} (Pareto) ---")
            print(top_k[["synset", "acc_clean", "acc_corrupt", "abs_drop"]])

    # print("HDBSCAN")
    # across_models_df = get_cross_model_df(fragile_dfs)
    # print(across_models_df.sort_values("fragile_count", ascending=False))
    # print(len(across_models_df))

    # run_clustering(across_models_df)


def _load_model_dfs_by_alias(
    models: list[str], alias: str, data_path: str
) -> dict[str, pd.DataFrame]:
    alexnet_df = get_alexnet_df_by_alias(alias, data_path)
    dfs = {}
    for model in models:
        try:
            dfs[model] = get_df_by_alias_with_rmce(model, alias, alexnet_df, data_path)
        except FileNotFoundError:
            pass
    return dfs


def _load_model_dfs_by_group(
    models: list[str], group: str, severity: int | None, data_path: str
) -> dict[str, pd.DataFrame]:
    from space import CorruptionVariations
    from constants import IMAGENET_C_SEVERITIES

    variation = CorruptionVariations(
        groups=[group],
        severities=[severity] if severity else IMAGENET_C_SEVERITIES,
    )
    all_dfs = get_dfs_for_all_models(variation, data_path)
    return {m: all_dfs[m] for m in models if m in all_dfs}


def _fragile_synsets(
    dfs: dict[str, pd.DataFrame],
    alexnet_df: pd.DataFrame,
    definition,
    select_top_k: int | None,
) -> tuple[list[set], dict[str, pd.DataFrame]]:
    fragile_sets: list[set] = []
    fragile_dfs: dict[str, pd.DataFrame] = {}
    for model, df in dfs.items():
        df = _get_fragile(df, alexnet_df, definition)
        candidates = df[df["is_strongly_fragile"] == 1]
        if select_top_k:
            candidates = select_top_k_fragile(candidates, select_top_k)
        fragile_sets.append(set(candidates["synset"]))
        fragile_dfs[model] = df
    return fragile_sets, fragile_dfs


def _agg_by_family(
    synsets: set | None, dfs: dict[str, pd.DataFrame], suffix: str
) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(list(dfs.values()))
    if synsets is not None:
        combined = combined[combined["synset"].isin(synsets)]
    agg_spec: dict = {
        f"acc_clean_{suffix}": ("acc_clean", "mean"),
        f"acc_corrupt_{suffix}": ("acc_corrupt", "mean"),
        f"rel_drop_{suffix}": ("rel_drop", "mean"),
    }
    if "y_true" in combined.columns:
        agg_spec["y_true"] = ("y_true", "first")
    return combined.groupby("synset").agg(**agg_spec).reset_index()


def _pareto_2d(df: pd.DataFrame, maximize_col: str, minimize_col: str) -> pd.Index:
    """Pareto front: maximize maximize_col, minimize minimize_col."""
    hi = df[maximize_col].values
    lo = df[minimize_col].values
    n = len(df)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if hi[j] >= hi[i] and lo[j] <= lo[i]:
                if hi[j] > hi[i] or lo[j] < lo[i]:
                    is_dominated[i] = True
                    break
    return df.index[~is_dominated]


def run_arch_contrast(args: argparse.Namespace) -> None:
    if not args.dataset and not args.group:
        raise ValueError("--arch-contrast requires --dataset or --group")

    vit_keys = args.vit_models
    cnn_keys = args.cnn_models
    label = args.dataset if args.dataset else f"{args.group} sev={args.severity or 'all'}"

    print(f"[arch-contrast] {label}")
    print(f"  ViT: {vit_keys}")
    print(f"  CNN: {cnn_keys}")

    if args.dataset:
        vit_dfs = _load_model_dfs_by_alias(vit_keys, args.dataset, args.data_path)
        cnn_dfs = _load_model_dfs_by_alias(cnn_keys, args.dataset, args.data_path)
    else:
        vit_dfs = _load_model_dfs_by_group(vit_keys, args.group, args.severity, args.data_path)
        cnn_dfs = _load_model_dfs_by_group(cnn_keys, args.group, args.severity, args.data_path)

    if not vit_dfs or not cnn_dfs:
        print("  Not enough data loaded.")
        return

    vit_agg = _agg_by_family(None, vit_dfs, "vit")
    cnn_agg = _agg_by_family(None, cnn_dfs, "cnn")
    df = vit_agg.merge(cnn_agg, on=["synset", "y_true"])

    df["delta_vit"] = df["rel_drop_vit"] - df["rel_drop_cnn"]
    df["delta_cnn"] = df["rel_drop_cnn"] - df["rel_drop_vit"]

    # ViT-exclusive: max rel_drop_vit, min rel_drop_cnn + twardy filtr delta + rel_drop_vit > 0
    vit_candidates = df[(df["delta_vit"] > args.delta) & (df["rel_drop_vit"] > 0) & (df["acc_clean_vit"] > 0.3)]
    vit_idx = _pareto_2d(vit_candidates, maximize_col="rel_drop_vit", minimize_col="rel_drop_cnn")
    vit_df = vit_candidates.loc[vit_idx].copy()

    # CNN-exclusive: max rel_drop_cnn, min rel_drop_vit + twardy filtr delta + rel_drop_cnn > 0
    cnn_candidates = df[(df["delta_cnn"] > args.delta) & (df["rel_drop_cnn"] > 0) & (df["acc_clean_cnn"] > 0.3)]
    cnn_idx = _pareto_2d(cnn_candidates, maximize_col="rel_drop_cnn", minimize_col="rel_drop_vit")
    cnn_df = cnn_candidates.loc[cnn_idx].copy()

    # usuń przecięcie — synset nie może być w obu zbiorach
    overlap = set(vit_df["synset"]) & set(cnn_df["synset"])
    if overlap:
        vit_df = vit_df[~vit_df["synset"].isin(overlap)]
        cnn_df = cnn_df[~cnn_df["synset"].isin(overlap)]

    if args.select_top_k:
        vit_df = vit_df.nlargest(args.select_top_k, "rel_drop_vit")
        cnn_df = cnn_df.nlargest(args.select_top_k, "rel_drop_cnn")

    cols = ["synset", "y_true", "rel_drop_vit", "rel_drop_cnn",
            "acc_clean_vit", "acc_corrupt_vit", "acc_clean_cnn", "acc_corrupt_cnn"]
    print(f"\n--- ViT-exclusive Pareto (↑ rel_drop_vit, ↓ rel_drop_cnn): {len(vit_df)} synsets ---")
    print(vit_df[[c for c in cols if c in vit_df.columns]].sort_values("rel_drop_vit", ascending=False).to_string())
    print(f"\n--- CNN-exclusive Pareto (↑ rel_drop_cnn, ↓ rel_drop_vit): {len(cnn_df)} synsets ---")
    print(cnn_df[[c for c in cols if c in cnn_df.columns]].sort_values("rel_drop_cnn", ascending=False).to_string())

    if args.save_tables:
        from .representation import arch_contrast_to_latex
        arch_contrast_to_latex(
            vit_df=vit_df,
            cnn_df=cnn_df,
            label=label,
            definition_name="pareto",
            save=True,
        )

    if args.scatter or args.save_tables:
        from .representation import arch_contrast_scatter
        arch_contrast_scatter(df, vit_df, cnn_df, label=label,
                              vit_keys=vit_keys, cnn_keys=cnn_keys)


def run_arch_contrast_v2(args: argparse.Namespace) -> None:
    from model import MODELS
    from .arch_contrast import compute_metrics, select_arch_fragile, plot_arch_contrast_scatter

    if not args.dataset and not args.group:
        raise ValueError("--arch-contrast-v2 requires --dataset or --group")

    vit_keys = args.vit_models
    cnn_keys = args.cnn_models
    label = args.dataset if args.dataset else f"{args.group} sev={args.severity or 'all'}"

    vit_label = MODELS[vit_keys[0]] if len(vit_keys) == 1 else "ViT (average)"
    cnn_label = MODELS[cnn_keys[0]] if len(cnn_keys) == 1 else "CNN (average)"

    print(f"[arch-contrast-v2] {label}  theta_a={args.theta_a}  theta_min={args.theta_min}")
    print(f"  ViT: {vit_keys}  CNN: {cnn_keys}")

    if args.dataset:
        vit_dfs = _load_model_dfs_by_alias(vit_keys, args.dataset, args.data_path)
        cnn_dfs = _load_model_dfs_by_alias(cnn_keys, args.dataset, args.data_path)
    else:
        vit_dfs = _load_model_dfs_by_group(vit_keys, args.group, args.severity, args.data_path)
        cnn_dfs = _load_model_dfs_by_group(cnn_keys, args.group, args.severity, args.data_path)

    if not vit_dfs or not cnn_dfs:
        print("  Not enough data loaded.")
        return

    vit_agg = _agg_by_family(None, vit_dfs, "vit")
    cnn_agg = _agg_by_family(None, cnn_dfs, "cnn")
    merged = vit_agg.merge(cnn_agg, on=["synset", "y_true"])

    df = merged.rename(columns={
        "acc_clean_vit": "acc_vit_clean",
        "acc_corrupt_vit": "acc_vit_corrupt",
        "acc_clean_cnn": "acc_cnn_clean",
        "acc_corrupt_cnn": "acc_cnn_corrupt",
    })

    vit_fragile, cnn_fragile, excluded = select_arch_fragile(
        df,
        theta_a=args.theta_a,
        theta_min=args.theta_min,
        apply_pareto=not args.no_pareto,
    )
    if not excluded.empty:
        print(f"\n  [excluded — negative drop] {len(excluded)} synsets skipped")

    cols = ["synset", "y_true", "d_vit", "d_cnn", "asymmetry_vit", "asymmetry_cnn", "delta_g",
            "acc_vit_clean", "acc_vit_corrupt", "acc_cnn_clean", "acc_cnn_corrupt"]
    print(f"\n--- {vit_label}-exclusive: {len(vit_fragile)} synsets ---")
    print(vit_fragile[[c for c in cols if c in vit_fragile.columns]].sort_values("d_vit", ascending=False).to_string())
    print(f"\n--- {cnn_label}-exclusive: {len(cnn_fragile)} synsets ---")
    print(cnn_fragile[[c for c in cols if c in cnn_fragile.columns]].sort_values("d_cnn", ascending=False).to_string())

    if args.scatter or args.save_tables:
        enriched = compute_metrics(df)
        plot_arch_contrast_scatter(
            enriched, vit_fragile, cnn_fragile,
            vit_label=vit_label,
            cnn_label=cnn_label,
            title=label,
        )


def run_dataset_model_intersection(args: argparse.Namespace) -> None:
    _MODELS = {
        "resnet50": "ResNet-50",
        "efficientnet_b4": "EfficientNet-B4",
        "vit_b_16": "ViT-B/16",
        "convnext_base": "ConvNeXt-Base",
    }

    if not args.dataset:
        raise ValueError("--dataset-intersection requires --dataset")

    definition = DEFINITIONS[args.definition]
    print(f"[dataset-intersection] dataset={args.dataset}  definition={definition.label}")

    alexnet_df = get_alexnet_df_by_alias(args.dataset, args.data_path)

    fragile_sets: list[set] = []
    all_dfs: list[pd.DataFrame] = []

    for model_key, model_label in _MODELS.items():
        try:
            df = get_df_by_alias_with_rmce(model_key, args.dataset, alexnet_df, args.data_path)
            df = _get_fragile(df, alexnet_df, definition)
            candidates = df[df["is_strongly_fragile"] == 1]
            if args.select_top_k:
                candidates = select_top_k_fragile(candidates, args.select_top_k)
            synsets = set(candidates["synset"])
            fragile_sets.append(synsets)
            all_dfs.append(df)
            print(f"  {model_label}: {len(synsets)} fragile synsets")
        except FileNotFoundError:
            print(f"  {model_label}: missing data, skipping")

    if not fragile_sets:
        print("No data loaded.")
        return

    intersection = set.intersection(*fragile_sets)
    print(f"\nIntersection: {len(intersection)} synsets fragile across all {len(fragile_sets)} models")

    if not intersection:
        print("Empty intersection.")
        return

    # combined = pd.concat(all_dfs)
    # combined = combined[combined["synset"].isin(intersection)]
    # agg = (
    #     combined.groupby("synset")
    #     .agg(
    #         y_true=("y_true", "first"),
    #         acc_clean=("acc_clean", "mean"),
    #         acc_corrupt=("acc_corrupt", "mean"),
    #         rel_drop=("rel_drop", "mean"),
    #         abs_drop=("abs_drop", "mean"),
    #         RmCE=("RmCE", "mean"),
    #     )
    #     .reset_index()
    # )

    # print(agg[["synset", "y_true", "acc_clean", "acc_corrupt", "rel_drop", "abs_drop"]].to_string())
    # print(f"\n{len(agg)} synsets")

    if args.save_tables:
        from .representation import dataset_intersection_to_latex
        dataset_intersection_to_latex(
            intersection,
            dataset=args.dataset,
            definition_name=args.definition,
            save=True,
        )


def run_sweep(args: argparse.Namespace):
    out = Path(args.output_dir)

    for exp_name, variations in EXPERIMENTS.items():
        print(f"\n[sweep] experiment: {exp_name}")
        dfs = get_dfs_for_all_models(variations, args.data_path)

        for def_name, definition in DEFINITIONS.items():
            print(f"  definition: {definition.label}")

            if args.model:
                df = _get_fragile(dfs[args.model], dfs["alexnet"], definition)
                fragile = df[df["is_strongly_fragile"] == 1]
                # df[df["is_strongly_fragile"] == 1].to_csv(
                # dest / "fragile_classes.csv", index=False
                # )
                if args.save_tables:
                    from .representation import to_latex

                    to_latex(
                        fragile,
                        save=True,
                        filename=f"{args.model}_{exp_name}_{def_name}.txt",
                        path=f"fragile/tables/{args.model}",
                        single_model=True,
                    )

                if args.select_top_k:
                    top_k = select_top_k_fragile(df, args.select_top_k)
                    print(f"\n--- Top {args.select_top_k} (Pareto) ---")
                    print(top_k[["synset", "acc_clean", "acc_corrupt", "abs_drop"]])

                    to_latex(
                        top_k,
                        save=True,
                        filename=f"{args.model}_{exp_name}_{def_name}.txt",
                        path=f"fragile/tables/{args.model}_pereto",
                        single_model=True,
                    )
            else:
                fragile_dfs = [
                    _get_fragile(df, dfs["alexnet"], definition) for df in dfs.values()
                ]
                cross = get_cross_model_fragile(fragile_dfs, definition, min_models=15)
                dest = out / "fragile" / exp_name / def_name
                dest.mkdir(parents=True, exist_ok=True)
                cross.to_csv(dest / "cross_model.csv", index=False)

                if args.save_tables:
                    from .representation import to_latex

                    to_latex(cross, save=True, filename=f"{exp_name}_{def_name}.txt")


def run_common(args: argparse.Namespace) -> None:
    definitions = (
        list(DEFINITIONS.values())
        if args.definition == "all"
        else [DEFINITIONS[args.definition]]
    )
    # out = Path(args.output_dir)

    for c in [4, 5, 6]:
        for definition in definitions:
            print(f"\n[common] definition: {definition.label}")
            common = get_common_fragile_across_experiments(
                definition, args.data_path, c
            )
            print(f"  {len(common)} synsets fragile across all experiments")
            print(common)

        # dest = out / "fragile" / "common_across_experiments"
        # dest.mkdir(parents=True, exist_ok=True)
        # pd.DataFrame({"synset": sorted(common)}).to_csv(
        # dest / f"{definition.name}.csv", index=False
        # )


def get_common_fragile_across_experiments(
    definition: FragileDefinition,
    data_path: str,
    min_experiments: int = 7,
) -> set[str]:
    counter = Counter()
    print("INTERSECTION MIN EXPERIMENTS: ", min_experiments)

    for exp_name, variations in EXPERIMENTS.items():
        dfs = get_dfs_for_all_models(variations, data_path)
        fragile_dfs = [
            _get_fragile(df, dfs["alexnet"], definition) for df in dfs.values()
        ]
        cross = get_cross_model_fragile(fragile_dfs, definition, min_models=15)
        counter.update(set(cross["synset"]))

    return {s for s, count in counter.items() if count >= min_experiments}


# def get_common_fragile_across_experiments(
#     definition: FragileDefinition,
# data_path: str,
# ) -> set[str]:
#     synset_sets = []
#     for exp_name, variations in EXPERIMENTS.items():
#         print(f"  [common] experiment: {exp_name}")
#         dfs = get_dfs_for_all_models(variations, data_path)
#         fragile_dfs = [_get_fragile(df, dfs["alexnet"], definition) for df in dfs.values()]
#         cross = get_cross_model_fragile(fragile_dfs, definition, min_models=15)
#         print(len(cross))
#         synset_sets.append(set(cross["synset"]))
#     if not synset_sets:
#         return set()
#     return set.intersection(*synset_sets)


def _get_fragile(df, alexnet_df, definition: FragileDefinition):
    df = calculate_relative_drop(df)
    df_a = get_absolute_fragile(df)
    df_b = get_relative_drop_fragile(df)
    df_c = get_rmce_fragile(df, alexnet_df)
    strong_fragile = get_strongly_fragile(df_a, df_b, df_c, definition)

    # print(df.head())
    super_giga_fragile = strong_fragile[strong_fragile["is_strongly_fragile"] == 1]
    # print(super_giga_fragile)
    # print(len(super_giga_fragile))

    return df.merge(strong_fragile, on="synset")


def identify_fragile_cluster(
    features: pd.DataFrame, labels: np.ndarray
) -> pd.DataFrame:
    df = features.copy()
    df["cluster"] = labels
    cluster_stats = df.groupby("cluster")[["acc_clean", "rel_drop"]].mean()
    eligible = cluster_stats[cluster_stats["acc_clean"] >= 0.65]
    fragile_cluster_id = eligible["rel_drop"].idxmax()
    print(
        f"  fragile cluster: {fragile_cluster_id}, stats: {cluster_stats.loc[fragile_cluster_id].to_dict()}"
    )
    return df[df["cluster"] == fragile_cluster_id][["synset"]]


def get_fragile_cluster_id(df: pd.DataFrame):
    cluster_stats = df.groupby("cluster")[["acc_clean", "rel_drop"]].mean()
    eligible = cluster_stats[cluster_stats["acc_clean"] >= 0.65]
    fragile_cluster_id = eligible["rel_drop"].idxmax()
    print(
        f"  Fragile cluster: {fragile_cluster_id}, stats: {cluster_stats.loc[fragile_cluster_id].to_dict()}"
    )

    return fragile_cluster_id


def run_clustering_fragile_sweep(args: argparse.Namespace):
    import numpy as np

    CLUSTERING_FEATURES = ["acc_clean", "acc_corrupt", "rel_drop", "abs_drop"]

    for exp_name, variations in EXPERIMENTS.items():
        print(f"\n[clustering sweep] experiment: {exp_name}")
        dfs = get_dfs_for_all_models(variations, args.data_path)

        if False:
            fragile_cluster_sets = []
            for model, df in dfs.items():
                features = df[["synset"] + CLUSTERING_FEATURES].dropna()
                labels = run_kmeans(features[CLUSTERING_FEATURES], k=5)
                fragile_cluster = identify_fragile_cluster(features, labels)
                fragile_cluster_sets.append(set(fragile_cluster["synset"]))

            counter = Counter()
            for s in fragile_cluster_sets:
                counter.update(s)

            intersection = {synset for synset, count in counter.items() if count >= 15}

            cross_df = get_cross_model_df(
                dfs,
                agg_cols=["acc_clean", "acc_corrupt", "rel_drop", "abs_drop", "RmCE"],
            )
            cross_df["fragile_count"] = (
                cross_df["synset"].map(counter).fillna(0).astype(int)
            )
            relevant = cross_df[cross_df["synset"].isin(intersection)]

            print(
                f"  {len(intersection)} synsets in fragile cluster across >= {15} models"
            )
            print(intersection)

            relevant.sort_values("fragile_count", ascending=False, inplace=True)

            if args.save_tables:
                from .representation import to_latex

                to_latex(
                    relevant,
                    save=True,
                    filename=f"{exp_name}_clustering.txt",
                    clustering=True,
                )

        for model, df in dfs.items():
            features = df[["synset"] + CLUSTERING_FEATURES].dropna()
            labels = run_kmeans(features[CLUSTERING_FEATURES], k=7)
            df["cluster"] = labels
            cluster_stats = df.groupby("cluster")[
                ["acc_clean", "acc_corrupt", "rel_drop", "abs_drop"]
            ].mean()
            # cluster_stats = cluster_stats.sort_values(["acc_clean", "rel_drop"], ascending=False)
            print(cluster_stats)
            cluster_stats_to_latex(cluster_stats, filename=f"{exp_name}_{model}")
            fragile_cluster_id = get_fragile_cluster_id(df)

            projected_df = run_pca(df)
            # projected_df = run_umap(projected_df)

            plot_kmeans(
                projected_df,
                projection="pca",
                filename=f"{exp_name}_{model}",
                fragile_cluster_id=fragile_cluster_id,
                output_path=f"images/clustering",
            )

        # print(cross_df[cross_df["cluster"] == 3]["synset"])

        # counter = Counter()
        # for s in fragile_cluster_sets:
        #     counter.update(s)

        # intersection = {
        #     synset for synset, count in counter.items()
        #     if count >= args.min_models
        # }
        # intersection = set.intersection(*fragile_cluster_sets)
        # intersection_df = pd.DataFrame({"synset": list(intersection)})
        # print(intersection_df)


def run_experiment_fragile_sweep(args: argparse.Namespace):
    if not args.model:
        raise ValueError("--exp-sweep requires --model")

    definition = DEFINITIONS[args.definition]
    rows = []

    for exp_name, variation in EXPERIMENTS.items():
        print(f"[exp-sweep] {exp_name}")
        try:
            alexnet_df = get_rmce_alexnet_df(variation, args.data_path)
            model_df = get_df_for_model(variation, args.model, args.data_path)
            df = _get_fragile(model_df, alexnet_df, definition)
            count = int(df["is_strongly_fragile"].sum())
        except FileNotFoundError:
            count = None

        rows.append({"experiment": exp_name, "fragile_count": count})

    result = pd.DataFrame(rows)
    print(result.set_index("experiment"))

    if args.save_tables:
        from .representation import experiment_sweep_to_latex
        experiment_sweep_to_latex(result, model=args.model, definition_name=args.definition)


def run_granular_fragile_sweep(args: argparse.Namespace):
    from space import CorruptionVariations
    from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES

    if not args.model:
        raise ValueError("--granular-sweep requires --model")

    definition = DEFINITIONS[args.definition]
    rows = []

    for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items():
        for corruption in corruptions:
            for severity in IMAGENET_C_SEVERITIES:
                variation = CorruptionVariations(
                    groups=[group],
                    corruptions=[corruption],
                    severities=[severity],
                )
                try:
                    alexnet_df = get_rmce_alexnet_df(variation, args.data_path)
                    model_df = get_df_for_model(variation, args.model, args.data_path)
                    df = _get_fragile(model_df, alexnet_df, definition)
                    count = int(df["is_strongly_fragile"].sum())
                except FileNotFoundError:
                    count = None

                rows.append({
                    "group": group,
                    "corruption": corruption,
                    "severity": severity,
                    "fragile_count": count,
                })

    result = pd.DataFrame(rows)
    print(result.pivot(index="corruption", columns="severity", values="fragile_count"))

    if args.save_tables:
        from .representation import granular_sweep_to_latex
        granular_sweep_to_latex(
            result,
            model=args.model,
            definition_name=args.definition,
        )


def run_granular_group_intersection_sweep(args: argparse.Namespace) -> None:
    from space import CorruptionVariations
    from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES

    if not args.model:
        raise ValueError("--granular-group-sweep requires --model")

    definition = DEFINITIONS[args.definition]

    for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items():
        print(f"\n[granular-group] {group}")
        fragile_sets: list[set] = []
        all_dfs: list[pd.DataFrame] = []

        for corruption in corruptions:
            for severity in IMAGENET_C_SEVERITIES:
                variation = CorruptionVariations(
                    groups=[group],
                    corruptions=[corruption],
                    severities=[severity],
                )
                try:
                    alexnet_df = get_rmce_alexnet_df(variation, args.data_path)
                    model_df = get_df_for_model(variation, args.model, args.data_path)
                    df = _get_fragile(model_df, alexnet_df, definition)
                    fragile_sets.append(set(df[df["is_strongly_fragile"] == 1]["synset"]))
                    all_dfs.append(df)
                except FileNotFoundError:
                    print(f"  missing: {corruption} sev{severity}")

        if not fragile_sets:
            print("  no data")
            continue

        intersection = set.intersection(*fragile_sets)
        print(f"  intersection size: {len(intersection)}")

        if not intersection:
            print("  empty intersection")
            continue

        combined = pd.concat(all_dfs)
        combined = combined[combined["synset"].isin(intersection)]
        agg = (
            combined.groupby("synset")
            .agg(
                y_true=("y_true", "first"),
                acc_clean=("acc_clean", "mean"),
                acc_corrupt=("acc_corrupt", "mean"),
                rel_drop=("rel_drop", "mean"),
                abs_drop=("abs_drop", "mean"),
                RmCE=("RmCE", "mean"),
            )
            .reset_index()
        )

        if args.select_top_k:
            agg = select_top_k_fragile(agg, args.select_top_k)

        print(agg[["synset", "acc_clean", "acc_corrupt", "rel_drop", "abs_drop"]].to_string())

        if args.save_tables:
            from .representation import to_latex
            to_latex(
                agg,
                save=True,
                filename=f"{args.model}_{group}.txt",
                path="fragile/granual",
                single_model=True,
            )


def run_robust_classes(args: argparse.Namespace) -> None:
    if not args.model:
        raise ValueError("--robust-classes requires --model")
    if not args.select_top_k:
        raise ValueError("--robust-classes requires --select-top-k")

    df = get_dfs_for_experiment(args.exp, args.model, args.data_path)
    top_k = select_top_k_robust(df, args.select_top_k)
    top_k = top_k.copy()
    top_k["y_true"] = top_k.get("y_true", 0) if "y_true" in top_k.columns else 0

    print(top_k[["synset", "acc_clean", "acc_corrupt", "rel_drop", "abs_drop"]].to_string())
    print(f"\n{len(top_k)} robust classes")

    if args.save_tables:
        from .representation import to_latex
        to_latex(
            top_k,
            save=True,
            filename=f"{args.model}_{args.exp}_robust_top{args.select_top_k}.txt",
            path="fragile/robust",
            single_model=True,
        )


def run_granular_group_cross_model_avg_sweep(args: argparse.Namespace) -> None:
    from .representation import generate_fragile_classes_table

    _MODELS = {
        "resnet50": "ResNet-50",
        "efficientnet_b4": "EfficientNet-B4",
        "vit_b_16": "ViT-B/16",
        "convnext_base": "ConvNeXt-Base",
    }
    _GROUPS = ["blur", "noise", "digital", "weather"]

    definition = DEFINITIONS[args.definition]
    data: dict[str, dict[str, set]] = {}

    for model_key in _MODELS:
        print(f"\n[granular-group-cross-model-avg] {model_key}")
        data[model_key] = {}

        for group in _GROUPS:
            try:
                df = get_dfs_for_experiment(group, model_key, args.data_path)
                alexnet_df = get_rmce_alexnet_df(EXPERIMENTS[group], args.data_path)
                df = _get_fragile(df, alexnet_df, definition)
                candidates = df[df["is_strongly_fragile"] == 1]
                if args.select_top_k:
                    candidates = select_top_k_fragile(candidates, args.select_top_k)
                fragile_synsets = set(candidates["synset"])
            except FileNotFoundError:
                fragile_synsets = set()

            data[model_key][group] = fragile_synsets
            print(f"  {group}: {len(fragile_synsets)} synsets")

    latex = generate_fragile_classes_table(
        data,
        model_labels=_MODELS,
        filename="cross_model_avg.txt",
        save=args.save_tables,
    )
    if not args.save_tables:
        print(latex)


def run_granular_group_cross_model_sweep(args: argparse.Namespace) -> None:
    from space import CorruptionVariations
    from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES
    from .representation import generate_fragile_classes_table

    _MODELS = {
        "resnet50": MODELS["resnet50"],
        "efficientnet_b4": MODELS["efficientnet_b4"],
        "vit_b_16": MODELS["vit_b_16"],
        "convnext_base": MODELS["convnext_base"],
    }

    definition = DEFINITIONS[args.definition]
    data: dict[str, dict[str, set]] = {}

    for model_key in _MODELS:
        print(f"\n[granular-group-cross-model] {model_key}")
        data[model_key] = {}

        for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items():
            fragile_sets: list[set] = []

            active_corruptions = [args.corruption] if args.corruption else corruptions
            active_severities = [args.severity] if args.severity else IMAGENET_C_SEVERITIES

            for corruption in active_corruptions:
                if corruption not in corruptions:
                    continue
                for severity in active_severities:
                    variation = CorruptionVariations(
                        groups=[group],
                        corruptions=[corruption],
                        severities=[severity],
                    )
                    try:
                        alexnet_df = get_rmce_alexnet_df(variation, args.data_path)
                        model_df = get_df_for_model(variation, model_key, args.data_path)
                        df = _get_fragile(model_df, alexnet_df, definition)
                        fragile_sets.append(set(df[df["is_strongly_fragile"] == 1]["synset"]))
                    except FileNotFoundError:
                        pass

            intersection = set.intersection(*fragile_sets) if fragile_sets else set()
            data[model_key][group] = intersection
            print(f"  {group}: {len(intersection)} synsets")

    if args.intersect_models:
        all_groups = {g for mk in _MODELS for g in data[mk]}
        for group in all_groups:
            common = set.intersection(*(data[mk].get(group, set()) for mk in _MODELS))
            print(f"  [intersect] {group}: {len(common)} synsets across all models")
            for model_key in _MODELS:
                data[model_key][group] = common

    parts = ["cross_model"]
    if args.corruption:
        parts.append(args.corruption)
    if args.severity:
        parts.append(f"sev{args.severity}")
    if args.intersect_models:
        parts.append("intersect")
    filename = "_".join(parts) + ".txt"

    latex = generate_fragile_classes_table(
        data,
        model_labels=_MODELS,
        filename=filename,
        save=args.save_tables,
    )
    if not args.save_tables:
        print(latex)


def get_fragile_by_rmce(args: argparse.Namespace):
    for exp_name, variations in EXPERIMENTS.items():
        dfs = get_dfs_for_all_models(variations, args.data_path)

        counter = Counter()
        for model, df in dfs.items():
            fragile = set(df[df["RmCE"] > 1.5]["synset"])
            counter.update(fragile)

        intersection = {synset for synset, count in counter.items() if count >= 15}

        cross_df = get_cross_model_df(
            dfs, agg_cols=["acc_clean", "acc_corrupt", "rel_drop", "abs_drop", "RmCE"]
        )
        cross_df["fragile_count"] = (
            cross_df["synset"].map(counter).fillna(0).astype(int)
        )
        relevant = cross_df[cross_df["synset"].isin(intersection)]
        relevant.sort_values("fragile_count", ascending=False, inplace=True)

        if args.save_tables:
            from .representation import to_latex

            to_latex(
                relevant, save=True, filename=f"{exp_name}_rmce.txt", clustering=False
            )


def run_fisher_matrix(args: argparse.Namespace) -> None:
    from scipy.stats import fisher_exact
    from space import CorruptionVariations
    from constants import IMAGENET_C_CORRUPTION_GROUPS

    definition = DEFINITIONS[args.definition]

    if args.corruption and args.severity:
        group = None
        for g, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items():
            if args.corruption in corruptions:
                group = g
                break
        if not group:
            raise ValueError(f"Unknown corruption: {args.corruption}")
        variations = CorruptionVariations(
            groups=[group],
            corruptions=[args.corruption],
            severities=[args.severity],
        )
        label = f"{args.corruption}_{args.severity}"
    else:
        variations = EXPERIMENTS[args.exp]
        label = args.exp

    print(f"[fisher-matrix] {label}, definition={definition.label}")
    print("Loading data for all models...")

    dfs = get_dfs_for_all_models(variations, args.data_path)
    model_keys = list(MODELS.keys())
    n_models = len(model_keys)

    fragile_vectors: dict[str, np.ndarray] = {}
    for model in model_keys:
        df = _get_fragile(dfs[model], dfs["alexnet"], definition)
        df_sorted = df.sort_values("synset")
        fragile_vectors[model] = df_sorted["is_strongly_fragile"].values.astype(bool)
        n_fragile = fragile_vectors[model].sum()
        print(f"  {model}: {n_fragile} fragile classes")

    p_matrix = np.ones((n_models, n_models))

    print("\nComputing Fisher exact test for all pairs...")
    for i, model_i in enumerate(model_keys):
        for j, model_j in enumerate(model_keys):
            if i == j:
                continue
            vi = fragile_vectors[model_i]
            vj = fragile_vectors[model_j]

            a = np.sum(vi & vj)
            b = np.sum(vi & ~vj)
            c = np.sum(~vi & vj)
            d = np.sum(~vi & ~vj)

            contingency = [[a, b], [c, d]]
            _, p_value = fisher_exact(contingency)
            p_matrix[i, j] = p_value

    print("\n" + "=" * 80)
    print("Fisher Exact Test p-value Matrix (20x20)")
    print("=" * 80)

    header = "".ljust(20) + "".join([m[:8].ljust(10) for m in model_keys])
    print(header)
    print("-" * len(header))

    for i, model_i in enumerate(model_keys):
        row = model_i[:18].ljust(20)
        for j in range(n_models):
            if i == j:
                row += "   -     "
            else:
                p = p_matrix[i, j]
                if p < 0.001:
                    row += f"{p:.2e} ".ljust(10)
                else:
                    row += f"{p:.4f}   "
        print(row)

    significant = np.sum((p_matrix < 0.05) & (p_matrix > 0)) // 2
    total_pairs = n_models * (n_models - 1) // 2
    print(f"\nSignificant pairs (p < 0.05): {significant}/{total_pairs}")


# def _get_fragile(model, variations, data_path):
#     df, alexnet_df = build_df_per_class(model, variations, data_path)

#     df_a = get_absolute_fragile(df)
#     df_b = get_relative_drop_fragile(df)
#     df_c = get_rmce_fragile(df, alexnet_df)

#     strong_fragile = get_strongly_fragile(df_a, df_b, df_c)
#     df = df.merge(strong_fragile, on="synset")

#     print(df.head())

#     super_giga_fragile = df[df["is_strongly_fragile"] == 1]
#     print(super_giga_fragile)
#     print(len(super_giga_fragile))

#     return df

# print("---")

# df_a_b = df[(df["is_fragile_a"] == 1) & (df["is_fragile_b"] == 1)][["synset", "acc_clean", "acc_corrupt", "rel_drop", "RmCE"]]

# print("a i b")
# print(df_a_b.head(5))
# print(df_a_b.describe())
# print(len(df_a_b))

# df_a_c = df[(df["is_fragile_a"] == 1) & (df["is_fragile_c"] == 1)][["synset", "acc_clean", "acc_corrupt", "rel_drop", "RmCE"]]

# print("a i c")
# print(df_a_c.head(5))
# print(df_a_c.describe())
# print(len(df_a_c))

# df_b_c = df[(df["is_fragile_b"] == 1) & (df["is_fragile_c"] == 1)][["synset", "acc_clean", "acc_corrupt", "rel_drop", "RmCE"]]
# print("b i c")
# print(df_b_c.head(5))
# print(df_b_c.describe())
# print(len(df_b_c))


# klasa która spada nieproporcjonalnie (B) ORAZ albo była dobra i się posypała (A) albo jest gorsza niż AlexNet (C) B ∩ (A ∪ C).

#         synset  y_true  acc_clean  acc_corrupt  rel_drop      RmCE  fragile_count
# 16   n01737021      58   0.834737     0.323228  0.618620  2.264649             19
# 127  n02971356     478   0.577895     0.246063  0.576744  4.957643             19
# 34   n01985128     124   0.768421     0.280547  0.645239  4.345668             19
# 149  n03141823     523   0.675789     0.314512  0.541183  2.393621             19
# 263  n04154565     784   0.543158     0.227411  0.584867  5.456464             19
# 213  n03793489     673   0.470526     0.208028  0.564571  2.163447             19
# 183  n03532672     600   0.455789     0.189811  0.593777  4.841850             19
# 361  n15075141     999   0.595789     0.183032  0.700778  3.165321             19
# 273  n04254120     804   0.718947     0.368632  0.496115  3.980861             19
# 270  n04208210     792   0.753684     0.338400  0.560751  2.331311             19
# 284  n04332243     828   0.693684     0.336435  0.519060  1.978854             19
# 349  n07860988     961   0.515789     0.140168  0.731853  2.537980             19
# 320  n04557648     898   0.629474     0.283214  0.552617  3.763692             19
# 316  n04548362     893   0.695789     0.298414  0.579739  2.098814             19
# 305  n04493381     876   0.402105     0.182049  0.546747  2.378128             19
# 238  n03995372     740   0.683158     0.294751  0.575201  2.954414             19
# 266  n04192698     787   0.700000     0.353179  0.500802  2.864711             19
# 275  n04263257     809   0.661053     0.079509  0.878785  2.262230             19
# 135  n03041632     499   0.642222     0.290311  0.558382  3.097809             18
# 141  n03089624     509   0.701111     0.317541  0.547778  2.505904             18
# 19   n01755581      67   0.838889     0.268533  0.684423  1.682796             18
# 14   n01698640      50   0.917778     0.377719  0.591222  1.129516             18
# 5    n01641577      30   0.824444     0.390237  0.526236  0.793895             18
# 255  n04116512     767   0.695556     0.340667  0.515748  1.914868             18
# 352  n07880968     965   0.905556     0.374089  0.590530  1.227217             18
# 288  n04357314     838   0.553333     0.271896  0.523382  2.605898             18
# 243  n04026417     748   0.608889     0.250341  0.593431  2.182720             18
# 227  n03908714     710   0.722222     0.336415  0.543577  2.075721             18
# 201  n03759954     650   0.655294     0.303827  0.546140  1.984940             17
# 217  n03832673     681   0.438824     0.126447  0.713386  2.632386             17
# 262  n04141975     778   0.801176     0.346180  0.579718  1.679365             17
# 324  n04597913     910   0.817647     0.325537  0.607460  1.992885             17
# 293  n04376876     845   0.721176     0.340063  0.537183  1.923521             17
# 231  n03938244     721   0.907059     0.330965  0.637074  1.602636             17
# 100  n02769748     414   0.494118     0.144486  0.709132  2.029594             17
# 175  n03481172     587   0.702353     0.302839  0.579371  1.963534             17
# 122  n02910353     464   0.603529     0.297271  0.513938  3.555636             17
# 188  n03633091     618   0.560000     0.195671  0.653495  2.754507             17
# 18   n01742172      61   0.846250     0.418800  0.507195  0.809973             16
# 13   n01697457      49   0.885000     0.377850  0.573879  1.252841             16
# 0    n01498041       6   0.906250     0.318917  0.647814  1.302484             16
# 192  n03666591     626   0.831250     0.394917  0.533737  1.724183             16
# 216  n03814906     679   0.940000     0.419317  0.554828  0.850049             16
# 267  n04200800     788   0.907500     0.385600  0.577278  1.246576             16
# 281  n04325704     824   0.703750     0.378050  0.461607  2.775852             16
# 98   n02730930     411   0.823750     0.359700  0.566429  1.510579             16
# 161  n03291819     549   0.625000     0.269117  0.574052  1.772327             16
# 180  n03498962     596   0.728000     0.377191  0.494722  2.332506             15
# 165  n03372029     558   0.569333     0.305671  0.467936  2.777341             15
# 159  n03255030     543   0.797333     0.421031  0.481407  2.191201             15
# 131  n03000134     489   0.828000     0.381049  0.540537  1.654557             15
# 113  n02840245     446   0.745333     0.374027  0.506057  1.936579             15
# 104  n02791270     424   0.736000     0.223929  0.700470  1.798002             15
# 304  n04479046     869   0.726667     0.358222  0.509030  1.722776             15
# 332  n07614500     928   0.724000     0.159129  0.782048  1.763752             15
# 55


# A and B not C
#         synset  y_true  acc_clean  acc_corrupt  rel_drop      RmCE  fragile_count
# 5    n01641577      30   0.824444     0.390237  0.526236  0.793895             18
# 13   n01698640      50   0.917778     0.377719  0.591222  1.129516             18
# 0    n01498041       6   0.906250     0.318917  0.647814  1.302484             16
# 12   n01697457      49   0.885000     0.377850  0.573879  1.252841             16
# 15   n01742172      61   0.846250     0.418800  0.507195  0.809973             16
# 151  n03814906     679   0.940000     0.419317  0.554828  0.850049             16
# 181  n04200800     788   0.907500     0.385600  0.577278  1.246576             16
# 232  n07880968     965   0.914667     0.412409  0.551002  1.159770             15
# 8


# A and B
#         synset  y_true  acc_clean  acc_corrupt  rel_drop      RmCE  fragile_count
# 5    n01641577      30   0.824444     0.390237  0.526236  0.793895             18
# 13   n01698640      50   0.917778     0.377719  0.591222  1.129516             18
# 275  n07880968     965   0.905556     0.374089  0.590530  1.227217             18
# 185  n03938244     721   0.907059     0.330965  0.637074  1.602636             17
# 0    n01498041       6   0.906250     0.318917  0.647814  1.302484             16
# 12   n01697457      49   0.885000     0.377850  0.573879  1.252841             16
# 174  n03814906     679   0.940000     0.419317  0.554828  0.850049             16
# 16   n01742172      61   0.846250     0.418800  0.507195  0.809973             16
# 211  n04200800     788   0.907500     0.385600  0.577278  1.246576             16
# 253  n04597913     910   0.848000     0.347360  0.591815  2.027430             15
# 10
