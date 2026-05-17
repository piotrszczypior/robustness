import argparse
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.model import MODELS

THRESHOLD_HIGH = 0.75
COLOR_NEG = "#888780"
COLOR_BLUE = "#378ADD"
COLOR_RED = "#C0392B"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.2,
        "grid.color": "#888780",
        "figure.dpi": 150,
    }
)


def discover_models(data_dir: Path) -> list[str]:
    found = []
    for f in data_dir.glob("*_imagenet.csv"):
        name = f.stem.removesuffix("_imagenet")
        if name in MODELS:
            found.append(name)
    return sorted(found)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["synset", "is_correct", "corruption", "severity"])


def per_class_accuracy(df: pd.DataFrame) -> pd.Series:
    return df.groupby("synset")["is_correct"].mean()


def compute_drops(acc_clean: pd.Series, acc_corrupt: pd.Series) -> pd.DataFrame:
    merged = pd.DataFrame({"acc_clean": acc_clean, "acc_corrupt": acc_corrupt}).dropna()
    merged["abs_drop"] = merged["acc_clean"] - merged["acc_corrupt"]
    merged["rel_drop"] = merged["abs_drop"] / merged["acc_clean"].replace(0, np.nan)
    return merged.dropna(subset=["rel_drop"])


def bar_color(mid: float) -> str:
    if mid < 0:
        return COLOR_NEG
    elif mid < THRESHOLD_HIGH:
        return COLOR_BLUE
    else:
        return COLOR_RED


def plot_histogram(
    rel_drop: pd.Series,
    title: str,
    output_path: Path,
    percentiles: dict,
    n_bins: int = 40,
):
    counts, edges = np.histogram(rel_drop, bins=n_bins)
    mids = (edges[:-1] + edges[1:]) / 2
    colors = [bar_color(m) for m in mids]

    n_total = len(rel_drop)
    n_neg = int((rel_drop < 0).sum())
    n_high = int((rel_drop >= THRESHOLD_HIGH).sum())

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = edges[1] - edges[0]
    ax.bar(mids, counts, width=bar_width * 0.95, color=colors, linewidth=0)

    ax.axvline(
        THRESHOLD_HIGH, color=COLOR_RED, linewidth=1.2, linestyle="--", alpha=0.8
    )

    for label, val in percentiles.items():
        ax.axvline(val, color="#444441", linewidth=1.0, linestyle=":", alpha=0.6)
        ax.text(val + 0.01, ax.get_ylim()[1] * 0.85, label, fontsize=8, color="#444441")

    ax.set_xlabel("relative accuracy drop", labelpad=8)
    ax.set_ylabel("number of classes", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="normal", pad=12)

    stats_text = f"n={n_total}  |  neg: {n_neg}  |  ≥50%: {n_high}"
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        color="#888780",
        ha="right",
        va="top",
    )

    p_text = "  ".join([f"{k}={v:.3f}" for k, v in percentiles.items()])
    ax.text(
        0.98,
        0.91,
        p_text,
        transform=ax.transAxes,
        fontsize=8,
        color="#444441",
        ha="right",
        va="top",
    )

    patches = [
        mpatches.Patch(color=COLOR_BLUE, label="drop < 75%"),
        mpatches.Patch(color=COLOR_RED, label="drop >= 75%"),
        mpatches.Patch(color=COLOR_NEG, label="negative drop"),
    ]
    ax.legend(handles=patches, fontsize=9, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def process_model(model: str, data_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = load_csv(data_dir / f"{model}_imagenet.csv")
    acc_clean = per_class_accuracy(baseline_df)

    corruption_files = sorted(
        f
        for f in data_dir.glob(f"{model}_imagenet_c_*.csv")
        if "embeddings" not in f.name
    )

    per_synset_drops: dict[str, list[float]] = {}

    for fpath in corruption_files:
        corrupt_df = load_csv(fpath)
        acc_corrupt = per_class_accuracy(corrupt_df)
        drops = compute_drops(acc_clean, acc_corrupt)
        for synset, row in drops.iterrows():
            per_synset_drops.setdefault(synset, []).append(row["rel_drop"])

    mean_drops = pd.Series(
        {synset: np.mean(vals) for synset, vals in per_synset_drops.items()},
        name="mean_rel_drop",
    ).dropna()

    percentiles = {
        "P75": float(np.percentile(mean_drops, 75)),
        "P90": float(np.percentile(mean_drops, 90)),
        "P95": float(np.percentile(mean_drops, 95)),
    }

    agg_title = f"{MODELS[model]} — aggregated (all corruptions)"
    plot_histogram(
        mean_drops,
        title=agg_title,
        output_path=output_dir / f"{model}_aggregated.png",
        percentiles=percentiles,
    )

    return {
        "model": MODELS[model],
        "model_id": model,
        "mean": float(mean_drops.mean()),
        "median": float(mean_drops.median()),
        "std": float(mean_drops.std()),
        **{f"p{k[1:]}": v for k, v in percentiles.items()},
        "n_fragile_75": int((mean_drops >= THRESHOLD_HIGH).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("images/histograms"))
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = discover_models(args.data_dir)
    if args.models:
        models = [m for m in models if m in args.models]

    print(f"Found {len(models)} models: {', '.join(models)}")

    summary_rows = []
    for model in models:
        print(f"\nProcessing {model}...")
        row = process_model(model, args.data_dir, args.output_dir)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("mean")
    cols_to_round = ["mean", "median", "std", "p75", "p90", "p95"]
    summary_df[cols_to_round] = summary_df[cols_to_round].round(3)

    summary_path = args.output_dir / "percentile_summary.csv"
    summary_df = summary_df.drop(columns=["model_id"])
    summary_df.to_latex(summary_path, index=False, float_format="%.3f")
    print(f"\nPercentile summary saved to {summary_path}")
    print(summary_df.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
