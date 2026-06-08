from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from fragile import read_df_for_model

from model import MODELS
from space import CorruptionVariations
from task import Task

TASK_NAME = "adv_acc_scatter"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="Scatter: ImageNet-C accuracy (x) vs adversarial accuracy (y) per class",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--attack", type=str, required=True, choices=["fgsm", "pgd"])
    parser.add_argument("--epsilon", type=int, default=4, help="epsilon")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--corruption", type=str, help="Specific corruption, e.g. shot_noise")
    group.add_argument("--group", type=str, help="Corruption group, e.g. noise (averages all)")
    parser.add_argument("--severity", type=int, default=3, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--adv-dir", type=str, default="adversarial")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="images/adversarial/scatter")
    parser.add_argument("--top-n", type=int, default=10, help="Number of classes to label")
    parser.add_argument("--robust", action="store_true", default=False)
    parser.add_argument("--heatmap", action="store_true", default=False)


def _eps_label(eps: float) -> str:
    return f"{eps}_255"


def _load_adv_acc(adv_dir: Path, model: str, attack: str, epsilon: float) -> pd.DataFrame:
    path = adv_dir / f"{model}_{attack}_{_eps_label(epsilon)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Adversarial CSV not found: {path}")
    df = pd.read_csv(path)
    acc = df.groupby("synset").agg(
        adv_acc=("is_correct", "mean"),
        class_name=("class_name", "first"),
    ).reset_index()
    return acc


def _load_corrupt_acc(model: str, 
                      group: str | None, 
                      corruption: str | None, 
                      severity: int) -> pd.DataFrame:
    
    if corruption:
        variation = CorruptionVariations(
            corruptions=[corruption],
            severities=[severity]
        )
        df = read_df_for_model(variation, model, definition="ab")
        label = f"{corruption.replace("_", " ").capitalize()} severity {severity}"
    else:
        variation = CorruptionVariations(
            groups=[group],
            severities=[severity]
        )
        df = read_df_for_model(variation, model, definition="ab")
        label = f"{group.replace("_", " ").capitalize()} severity {severity}"


    return df, label


def get_full_df(model: str):
    df = read_df_for_model(CorruptionVariations(), model, definition="ab")
    return df


def extend_with_adv_rel_drop(df):
    df["adv_rel_drop"] = (df["acc_clean"] - df["adv_acc"]) / df["acc_clean"]
    return df


def _build_adv_spearman_matrix(adv: pd.DataFrame, model: str) -> pd.DataFrame:
    from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES

    data: dict[int, dict[str, float]] = {}
    for severity in IMAGENET_C_SEVERITIES:
        row: dict[str, float] = {}
        for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items():
            for corruption in corruptions:
                try:
                    variation = CorruptionVariations(
                        groups=[group], corruptions=[corruption], severities=[severity]
                    )
                    df = read_df_for_model(variation, model, definition="ab")
                    merged = adv.merge(df[["synset", "acc_clean", "rel_drop"]], on="synset", how="inner")
                    if len(merged) < 5:
                        row[corruption] = float("nan")
                        continue
                    adv_rel_drop = (merged["acc_clean"] - merged["adv_acc"]) / merged["acc_clean"]
                    rho, _ = spearmanr(merged["rel_drop"], adv_rel_drop)
                    row[corruption] = float(rho)
                except FileNotFoundError:
                    row[corruption] = float("nan")
        data[severity] = row

    return pd.DataFrame(data).T

def run(args: argparse.Namespace) -> None:
    from .plot import plot_scatter, plot_scatter_robust

    adv_dir = Path(args.adv_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    adv = _load_adv_acc(adv_dir, args.model, args.attack, args.epsilon)
    df, corruption_label = _load_corrupt_acc(
        args.model, args.group, args.corruption, args.severity
    )

    df = adv.merge(df, on="synset", how="inner")
    df = extend_with_adv_rel_drop(df)
    if df.empty:
        print("ERROR: no common synsets after merge", file=sys.stderr)
        return

    print(f"Plotting {len(df)} classes  (adv={len(adv)}, corrupt={len(df)}, common={len(df)})")

    if args.heatmap:
        matrix = _build_adv_spearman_matrix(adv, args.model)
        eps_label = f"{args.epsilon}_255"
        out_heatmap = (
            output_dir / f"{args.model}_{args.attack}_{eps_label}_spearman_adv.png"
        )
        out_heatmap.parent.mkdir(parents=True, exist_ok=True)
        from .plot import heatmap as render_heatmap
        render_heatmap(
            matrix,
            out_heatmap,
            title=f"Spearman ρ ·  {MODELS[args.model]}  ·  {args.attack.upper()} ε={args.epsilon}/255",
        )
        return 

    if args.robust:
        plot_scatter_robust(
            df=df,
            model=args.model,
            attack=args.attack,
            epsilon=args.epsilon,
            corruption_label=corruption_label,
            output_dir=output_dir,
        )
        return 

    plot_scatter(
        df=df,
        model=args.model,
        attack=args.attack,
        epsilon=args.epsilon,
        corruption_label=corruption_label,
        output_dir=output_dir,
    )

