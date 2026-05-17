from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    from model import MODELS
    from utils import get_synset_to_label_imagenet1k
except ImportError:
    MODELS = None
    get_synset_to_label_imagenet1k = None


def load_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def per_class_accuracy(df: pd.DataFrame) -> pd.Series:
    return df.groupby("y_true")["is_correct"].mean()


def discover_models(data_dir: Path) -> list[str]:
    suffixes = {"_imagenet.csv", "_imagenet_r.csv"}
    candidates: dict[str, set] = {}
    for f in data_dir.glob("*.csv"):
        for suf in suffixes:
            if f.name.endswith(suf):
                prefix = f.name[: -len(suf)]
                candidates.setdefault(prefix, set()).add(suf)
    return [p for p, s in candidates.items() if len(s) == 2]


def build_results_table(
    data_dir: Path,
    models: list[str],
    synset_map: Optional[dict] = None,
    top_n: int = 20,
    metric: str = "relative",
) -> pd.DataFrame:
    per_model: list[pd.DataFrame] = []

    for model in models:
        try:
            df_clean = load_csv(data_dir, f"{model}_imagenet.csv")
            df_r = load_csv(data_dir, f"{model}_imagenet_r.csv")
        except FileNotFoundError as e:
            print(f"[SKIP] {model}: {e}")
            continue

        acc_clean = per_class_accuracy(df_clean).rename("acc_clean")
        acc_r = per_class_accuracy(df_r).rename("acc_r")

        merged = pd.concat([acc_clean, acc_r], axis=1).dropna()
        merged["model"] = model

        if metric == "absolute":
            merged["drop"] = merged["acc_clean"] - merged["acc_r"]
        elif metric == "relative":
            merged["drop"] = (merged["acc_clean"] - merged["acc_r"]) / merged[
                "acc_clean"
            ].clip(lower=1e-6)
        elif metric == "r_only":
            merged["drop"] = 1.0 - merged["acc_r"]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        per_model.append(merged.reset_index())

    if not per_model:
        raise RuntimeError("No model data loaded – check data_dir and file names.")

    all_data = pd.concat(per_model, ignore_index=True)

    agg = (
        all_data.groupby("y_true")
        .agg(
            acc_clean_mean=("acc_clean", "mean"),
            acc_clean_std=("acc_clean", "std"),
            acc_r_mean=("acc_r", "mean"),
            acc_r_std=("acc_r", "std"),
            drop_mean=("drop", "mean"),
            drop_std=("drop", "std"),
            n_models=("model", "count"),
        )
        .reset_index()
        .sort_values("drop_mean", ascending=False)
        .head(top_n)
    )

    if synset_map:
        agg["class_name"] = (
            agg["y_true"].map(synset_map).fillna(agg["y_true"].astype(str))
        )
    else:
        agg["class_name"] = agg["y_true"].astype(str)

    return agg


def format_latex(df: pd.DataFrame, metric: str, caption_extra: str = "") -> str:
    metric_labels = {
        "absolute": r"$\Delta$ acc (abs)",
        "relative": r"$\Delta$ acc (rel \%)",
        "r_only": r"Error on $\mathcal{R}$",
    }
    drop_label = metric_labels.get(metric, "Drop")

    display = pd.DataFrame()
    display["Class"] = df["class_name"]
    display["Clean acc"] = df["acc_clean_mean"].map(lambda x: f"{x:.3f}")
    display["R acc"] = df["acc_r_mean"].map(lambda x: f"{x:.3f}")

    if metric == "relative":
        display[drop_label] = df["drop_mean"].map(lambda x: f"{x * 100:.1f}")
    else:
        display[drop_label] = df["drop_mean"].map(lambda x: f"{x:.3f}")

    display[r"$\pm$"] = df["drop_std"].map(lambda x: f"{x:.3f}")
    display["N models"] = df["n_models"].astype(int)

    caption = (
        f"Top-{len(df)} hardest ImageNet-R classes ranked by mean {drop_label} "
        f"across {df['n_models'].iloc[0]} models. " + caption_extra
    )

    return display.to_latex(
        index=False,
        escape=False,
        column_format="l" + "r" * (len(display.columns) - 1),
        caption=caption,
        label="tab:worst_classes",
        position="t",
    )


def main():
    parser = argparse.ArgumentParser(description="Find hardest ImageNet-R classes.")
    parser.add_argument("--data_dir", type=Path, default=Path("results"))
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument(
        "--metric", choices=["absolute", "relative", "r_only"], default="relative"
    )
    parser.add_argument("--out_csv", type=Path, default=None)
    args = parser.parse_args()

    models = MODELS if MODELS is not None else discover_models(args.data_dir)
    if not models:
        raise RuntimeError(f"No models found in {args.data_dir}")
    print(f"Found {len(models)} models: {models}")

    synset_map = (
        get_synset_to_label_imagenet1k() if get_synset_to_label_imagenet1k else None
    )

    table = build_results_table(
        data_dir=args.data_dir,
        models=models,
        synset_map=synset_map,
        top_n=args.top_n,
        metric=args.metric,
    )

    latex = format_latex(table, metric=args.metric)
    print("\n" + "=" * 72)
    print(latex)
    print("=" * 72)

    if args.out_csv:
        table.to_csv(args.out_csv, index=False)
        print(f"\nSaved full table to {args.out_csv}")


if __name__ == "__main__":
    main()
