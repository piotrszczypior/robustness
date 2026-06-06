from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from constants import IMAGENET_R_SYNSETS

PRED_DIR = ROOT / "results"
OUT_DIR = ROOT / "results" / "representations" / "imagenet_r_synset_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "resnet50", "resnet152", "regnet_y_16gf", "resnext101_64x4d",
    "wide_resnet50_2", "wide_resnet101_2", "efficientnet_b4",
    "efficientnet_v2_m", "vit_b_16", "vit_l_16", "swin_b",
    "swin_v2_b", "maxvit_t", "convnext_base", "convnext_large",
]

TOP_MISTAKES = 5


def load_label_maps() -> tuple[dict[str, str], dict[int, str]]:
    with open(ROOT / "imagenet_class_index.json") as f:
        index = json.load(f)
    synset_to_label = {syn: label for syn, label in index.values()}
    idx_to_synset = {int(i): syn for i, (syn, _) in index.items()}
    return synset_to_label, idx_to_synset


def synset_acc(df: pd.DataFrame, synset: str) -> float | None:
    rows = df[df["synset"] == synset]
    if rows.empty:
        return None
    return float(rows["is_correct"].mean())


def common_mistakes(
    df_r: pd.DataFrame,
    synset: str,
    idx_to_synset: dict[int, str],
    synset_to_label: dict[str, str],
    top_n: int = TOP_MISTAKES,
) -> list[tuple[str, float]]:
    """Top-N wrong predicted labels (with fraction) for a given synset on ImageNet-R."""
    wrong = df_r[(df_r["synset"] == synset) & (df_r["is_correct"] == 0)]
    if wrong.empty:
        return []
    total = len(df_r[df_r["synset"] == synset])
    counts = wrong["y_pred"].value_counts().head(top_n)
    result = []
    for idx, cnt in counts.items():
        syn = idx_to_synset.get(int(idx), str(idx))
        label = synset_to_label.get(syn, syn)
        result.append((label, round(cnt / total, 3)))
    return result


def fmt_acc(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "n/a"


def mistakes_str(mistakes: list[tuple[str, float]]) -> str:
    if not mistakes:
        return "-"
    return ", ".join(f"{label} ({frac:.3f})" for label, frac in mistakes)


def mistakes_latex(mistakes: list[tuple[str, float]]) -> str:
    if not mistakes:
        return "--"
    parts = [rf"\textit{{{label}}} ({frac:.3f})" for label, frac in mistakes]
    return "; ".join(parts)


def write_txt(path: Path, model: str, rows: list[dict], synset_to_label: dict[str, str]) -> None:
    col_w = {"synset": 12, "label": 28, "clean": 9, "r_acc": 9, "common_mistakes": 60}
    header = (
        f"{'synset':<{col_w['synset']}}"
        f"{'label':<{col_w['label']}}"
        f"{'clean':>{col_w['clean']}}"
        f"{'r_acc':>{col_w['r_acc']}}"
        f"  common_mistakes"
    )
    sep = "-" * (sum(col_w.values()) + 2)
    lines = [f"Model: {model}", "", header, sep]
    for r in rows:
        label = synset_to_label.get(r["synset"], r["synset"])
        line = (
            f"{r['synset']:<{col_w['synset']}}"
            f"{label:<{col_w['label']}}"
            f"{r['clean']:>{col_w['clean']}}"
            f"{r['r_acc']:>{col_w['r_acc']}}"
            f"  {r['mistakes_str']}"
        )
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")


def write_latex(path: Path, model: str, rows: list[dict], synset_to_label: dict[str, str]) -> None:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        rf"\caption{{Per-class accuracy and common mistakes on ImageNet-R — \texttt{{{model}}}}}",
        rf"\label{{tab:imagenet_r__{model}}}",
        r"\begin{tabular}{llrrp{6cm}}",
        r"\toprule",
        r"Synset & Label & Clean acc & R acc & Common mistakes \\",
        r"\midrule",
    ]
    for r in rows:
        label = synset_to_label.get(r["synset"], r["synset"]).replace("_", r"\_")
        synset = r["synset"]
        lines.append(
            f"{synset} & {label} & {r['clean']} & {r['r_acc']} & {r['mistakes_latex']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def process_model(
    model: str,
    synset_to_label: dict[str, str],
    idx_to_synset: dict[int, str],
) -> None:
    clean_path = PRED_DIR / f"{model}_imagenet.csv"
    r_path = PRED_DIR / f"{model}_imagenet_r.csv"

    if not clean_path.exists() or not r_path.exists():
        print(f"  [skip] {model}: missing CSV(s)")
        return

    df_clean = pd.read_csv(clean_path)
    df_r = pd.read_csv(r_path)

    rows = []
    for synset in sorted(IMAGENET_R_SYNSETS):
        clean = fmt_acc(synset_acc(df_clean, synset))
        r_acc = fmt_acc(synset_acc(df_r, synset))
        mistakes = common_mistakes(df_r, synset, idx_to_synset, synset_to_label)
        rows.append({
            "synset": synset,
            "clean": clean,
            "r_acc": r_acc,
            "mistakes_str": mistakes_str(mistakes),
            "mistakes_latex": mistakes_latex(mistakes),
        })

    # Sort by ImageNet-R accuracy ascending (worst first)
    rows.sort(key=lambda r: float(r["r_acc"]) if r["r_acc"] != "n/a" else 1.0)

    write_txt(OUT_DIR / f"{model}.txt", model, rows, synset_to_label)
    write_latex(OUT_DIR / f"{model}.tex", model, rows, synset_to_label)
    print(f"  {model}: {len(rows)} classes written")


def main() -> None:
    synset_to_label, idx_to_synset = load_label_maps()

    print(f"ImageNet-R synset stats → {OUT_DIR}\n")
    for model in MODELS:
        process_model(model, synset_to_label, idx_to_synset)

    print("\nDone.")


if __name__ == "__main__":
    main()
