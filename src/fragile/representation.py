import re
from utils import get_synset_to_label_imagenet1k
import pandas as pd
from pathlib import Path


def to_latex(
    df: pd.DataFrame,
    save: bool = False,
    filename: str = "fragile_classes.tex",
    path="fragile/tables",
    clustering=False,
) -> None:
    df = df.copy()
    synset_to_label = get_synset_to_label_imagenet1k()
    df["label"] = df["synset"].map(lambda s: _map_synset(s, synset_to_label))
    df = df.rename(
        columns={
            "y_true": "index",
            "acc_clean": "accuracy clean",
            "acc_corrupt": "accuracy corrupt",
            "rel_drop": "relative drop",
            "abs_drop": "absolute drop",
            "RmCE": "RmCE",
            "fragile_count": "model count",
        }
    )
    cols = [
        "synset",
        "index",
        "label",
        "accuracy clean",
        "accuracy corrupt",
        "relative drop",
        "absolute drop",
        "RmCE",
        "model count",
    ]
    # latex = df[cols].to_latex(index=False, float_format="%.3f")
    latex = _to_table(df)

    if save:
        output = Path(path) if not clustering else Path(path) / "clustering"
        output.mkdir(exist_ok=True, parents=True)
        (output / filename).write_text(latex)
        print(f"Saved to {output}")
        return

    print(latex)

def _map_synset(synset, synset_to_label: dict[str, str]):
    label = synset_to_label[synset]
    return label.replace("_", " ").capitalize()
    

def _to_table(df: pd.DataFrame):
    header = (
        "\\toprule\n"
        " & & & \\multicolumn{2}{c}{Accuracy} & \\multicolumn{2}{c}{Drop} & & \\\\\n"
        "\\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n"
        "Synset & Index & Label & Clean & Corrupt & Relative & Absolute & RmCE & Count \\\\\n"
        "\\midrule\n"
    )

    rows = ""
    for _, row in df.iterrows():
        rows += " & ".join([
            str(row["synset"]),
            str(int(row["index"])),
            str(row["label"]),
            f"{row['accuracy clean']:.3f}",
            f"{row['accuracy corrupt']:.3f}",
            f"{row['relative drop']:.3f}",
            f"{row['absolute drop']:.3f}",
            f"{row['RmCE']:.3f}",
            str(int(row["model count"])),
        ]) + " \\\\\n"

    latex = (
        "\\begin{table}[H]\n"
        "\\caption{Caption}\n"
        "\\centering\n"
        "\\resizebox{\\textwidth}{!}{\n"
        "\\begin{tabular}{llrccccrc}\n"
        + header + rows +
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        "\\label{tab:fragile_classes_ab}\n"
        "\\end{table}\n"
    )

    return latex



def cluster_stats_to_latex(  df: pd.DataFrame,
    save: bool = False,
    filename: str = "cluster_stats.tex",
    path="fragile/tables/stats"):
    Path(path).mkdir(parents=True, exist_ok=True)
    output = Path(path) / filename
    df.to_latex(output)
