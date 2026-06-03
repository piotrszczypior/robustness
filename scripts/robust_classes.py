from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from fragile.experiments import EXPERIMENTS, get_df_for_model
from model import MODELS
from utils import get_synset_to_label_imagenet1k


def pareto_front_mask(df: pd.DataFrame) -> pd.Series:
    clean = df["acc_clean"].to_numpy()
    corrupt = df["acc_corrupt"].to_numpy()
    on_front = []
    for i in range(len(df)):
        dominated = any(
            (clean[j] >= clean[i] and corrupt[j] >= corrupt[i])
            and (clean[j] > clean[i] or corrupt[j] > corrupt[i])
            for j in range(len(df)) if j != i
        )
        on_front.append(not dominated)
    return pd.Series(on_front, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top/bottom robust classes for a single model on an ImageNet-C experiment"
    )
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--data-path", default="results")
    parser.add_argument("--exp", default="all_corruptions", choices=list(EXPERIMENTS))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--pareto", action="store_true", help="Show Pareto front (max clean + corrupt)")
    args = parser.parse_args()

    df = get_df_for_model(EXPERIMENTS[args.exp], args.model, args.data_path)

    label_map = get_synset_to_label_imagenet1k()
    df = df.copy()
    df["label"] = df["synset"].map(label_map).fillna(df["synset"])

    model_label = MODELS[args.model]
    print(f"\nModel: {model_label}  |  exp={args.exp}  |  {len(df)} classes\n")

    cols = ["synset", "label", "acc_clean", "acc_corrupt"]

    if args.pareto:
        df["on_front"] = pareto_front_mask(df)
        front = (
            df[df["on_front"]][cols]
            .sort_values("acc_corrupt", ascending=False)
        )
        print(f"=== PARETO FRONT ({len(front)} classes) ===")
        print(front.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return

    n = args.top_n
    df = df.sort_values("acc_corrupt", ascending=False).reset_index(drop=True)

    print(f"=== MOST ROBUST (top {n}) ===")
    print(df.head(n)[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n=== LEAST ROBUST (bottom {n}) ===")
    print(df.tail(n).iloc[::-1][cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
