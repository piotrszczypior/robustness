from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import matplotlib.pyplot as plt

from fragile.experiments import EXPERIMENTS, get_dfs_for_all_models
from model import MODELS
from utils import get_synset_to_label_imagenet1k


def pareto_front(df: pd.DataFrame) -> pd.Series:
    """Returns boolean mask of Pareto-optimal rows (maximise both objectives)."""
    n = len(df)
    clean = df["acc_clean"].to_numpy()
    corrupt = df["acc_corrupt"].to_numpy()
    on_front = []
    for i in range(n):
        dominated = any(
            (clean[j] >= clean[i] and corrupt[j] >= corrupt[i])
            and (clean[j] > clean[i] or corrupt[j] > corrupt[i])
            for j in range(n) if j != i
        )
        on_front.append(not dominated)
    return pd.Series(on_front, index=df.index)


def aggregate_per_model(
    dfs: dict[str, pd.DataFrame], synset: str | None
) -> pd.DataFrame:
    rows = []
    for model_key, df in dfs.items():
        if synset:
            df = df[df["synset"] == synset]
        if df.empty:
            continue
        rows.append({
            "model": model_key,
            "label": MODELS.get(model_key, model_key),
            "acc_clean": df["acc_clean"].mean(),
            "acc_corrupt": df["acc_corrupt"].mean(),
        })
    return pd.DataFrame(rows)


def render(agg: pd.DataFrame, output: Path) -> None:
    on_front = agg["on_front"]
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")

    # Draw Pareto front line (sorted by acc_clean)
    front = agg[on_front].sort_values("acc_clean")
    ax.step(
        front["acc_clean"], front["acc_corrupt"],
        where="post", color="#e05c2a", linewidth=1.4, linestyle="--", zorder=2,
    )

    # All models
    ax.scatter(
        agg.loc[~on_front, "acc_clean"],
        agg.loc[~on_front, "acc_corrupt"],
        color="#aaaaaa", s=60, zorder=3,
    )
    ax.scatter(
        agg.loc[on_front, "acc_clean"],
        agg.loc[on_front, "acc_corrupt"],
        color="#e05c2a", s=80, zorder=4,
    )

    for _, row in agg.iterrows():
        ax.annotate(
            row["label"],
            (row["acc_clean"], row["acc_corrupt"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7.5,
            color="#333333" if not row["on_front"] else "#c03a10",
        )

    ax.set_xlabel("Clean accuracy (mean per-class)", fontsize=11)
    ax.set_ylabel("Corrupt accuracy (mean per-class)", fontsize=11)
    ax.set_title("Pareto front: clean vs. corrupt accuracy", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pareto front: maximise clean and corrupt accuracy across models"
    )
    parser.add_argument("--data-path", default="results")
    parser.add_argument("--exp", default="all_corruptions", choices=list(EXPERIMENTS))
    parser.add_argument("--synset", default=None, help="Filter to a single synset")
    parser.add_argument("--output", type=Path, default=None, help="Optional plot output path")
    args = parser.parse_args()

    variations = EXPERIMENTS[args.exp]
    dfs = get_dfs_for_all_models(variations, args.data_path)

    agg = aggregate_per_model(dfs, args.synset)
    if agg.empty:
        print("No data found.")
        return

    agg["on_front"] = pareto_front(agg)

    label_map = get_synset_to_label_imagenet1k()
    header = f"exp={args.exp}"
    if args.synset:
        label = label_map.get(args.synset, args.synset)
        header += f"  synset={args.synset} ({label})"
    print(f"\n{header}\n")

    display = agg[["label", "acc_clean", "acc_corrupt", "on_front"]].copy()
    display = display.sort_values("acc_clean", ascending=False)
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    front = agg[agg["on_front"]].sort_values("acc_clean", ascending=False)
    print(f"\nPareto front ({len(front)} models):")
    print(front[["label", "acc_clean", "acc_corrupt"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.output:
        render(agg, args.output)


if __name__ == "__main__":
    main()
