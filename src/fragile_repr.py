from __future__ import annotations
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

METRICS_PATH = Path("results/representations/resnet50_class_metrics.parquet")
PRED_DIR = Path("embeddings")
OUT_DIR = Path("results/representations/fragility")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "resnet50"
METRIC = "angular_distance_median"
CORRUPTION = "elastic_transform"
SEVERITY = 3
K = 20

from utils import get_synset_to_label_imagenet1k


def clean_pred_path():
    return PRED_DIR / f"{MODEL}_imagenet_embeddings.parquet"


def cond_pred_path(corruption, severity):
    return PRED_DIR / f"{MODEL}_imagenet_c_digital_{corruption}_{severity}_embeddings.parquet"


def condition_table(df, corruption, severity):
    sub = df[(df["model"] == MODEL) & (df["corruption"] == corruption) & (df["severity"] == severity)]
    return sub.pivot_table(index="synset", columns="metric", values="value")


def per_class_accuracy(path):
    p = pd.read_parquet(path, columns=["synset", "y_true", "y_pred"])
    correct = (p["y_true"] == p["y_pred"]).astype(int)
    return correct.groupby(p["synset"]).mean()


def add_accuracy(table, corruption, severity):
    cp, op = clean_pred_path(), cond_pred_path(corruption, severity)
    if not (cp.exists() and op.exists()):
        print(f"(accuracy skipped: predictions not found in {PRED_DIR})")
        return table, None
    drop = (per_class_accuracy(cp) - per_class_accuracy(op)).reindex(table.index)
    print(drop)
    table = table.copy()
    table["acc_drop"] = drop
    rho = spearmanr(table[METRIC], table["acc_drop"], nan_policy="omit")[0]
    print(rho)
    return table, rho


def main():
    df = pd.read_parquet(METRICS_PATH)
    table = condition_table(df, CORRUPTION, SEVERITY)
    if METRIC not in table.columns:
        raise ValueError(f"metric {METRIC} not found")
    table = table.sort_values(METRIC, ascending=False)
    table, rho = add_accuracy(table, CORRUPTION, SEVERITY)

    print(f"{MODEL}  {CORRUPTION}  severity {SEVERITY}  ranked by {METRIC}")
    if rho is not None:
        print(f"validation: Spearman({METRIC}, acc_drop) = {rho:.3f}  (n={int(table['acc_drop'].notna().sum())})")


    synset_to_label = get_synset_to_label_imagenet1k()
    table["label"] = [synset_to_label[i] for i in table.index]

    cols = [METRIC] + [c for c in ["relative_shift_median", "acc_drop", "label"] if c in table.columns]
    print(f"\nTOP {K} most fragile:")
    print(table[cols].head(K).round(3).to_string())
    print(f"\nBOTTOM {K} most robust:")
    print(table[cols].tail(K).round(3).to_string())

    print(table[table.index == 'n12057211'])

    print(table.columns)

    out = OUT_DIR / f"fragility_{MODEL}_{CORRUPTION}_s{SEVERITY}.csv"
    table[cols].to_csv(out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()