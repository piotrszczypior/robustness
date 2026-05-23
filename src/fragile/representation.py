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
    single_model=True,
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

    if save:
        output = Path(path) if not clustering else Path(path) / "clustering"
        output.mkdir(exist_ok=True, parents=True)
        latex = _to_table(df, single_model=single_model)
        (output / filename).write_text(latex)
        print(f"Saved to {output}")
        return

    print(_to_table(df, single_model=single_model))


def _map_synset(synset, synset_to_label: dict[str, str]):
    label = synset_to_label[synset]
    return label.replace("_", " ").capitalize()


def _to_table(df: pd.DataFrame, single_model: bool = True):
    if single_model:
        header = (
            "\\toprule\n"
            " & & & \\multicolumn{2}{c}{Accuracy} & \\multicolumn{2}{c}{Drop} & \\\\\n"
            "\\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n"
            "Synset & Index & Label & Clean & Corrupt & Relative & Absolute & RmCE \\\\\n"
            "\\midrule\n"
        )
    else:
        header = (
            "\\toprule\n"
            " & & & \\multicolumn{2}{c}{Accuracy} & \\multicolumn{2}{c}{Drop} & & \\\\\n"
            "\\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n"
            "Synset & Index & Label & Clean & Corrupt & Relative & Absolute & RmCE & Count \\\\\n"
            "\\midrule\n"
        )

    rows = ""
    for _, row in df.iterrows():
        cols = [
            str(row["synset"]),
            str(int(row["index"])),
            str(row["label"]),
            f"{row['accuracy clean']:.3f}",
            f"{row['accuracy corrupt']:.3f}",
            f"{row['relative drop']:.3f}",
            f"{row['absolute drop']:.3f}",
            f"{row['RmCE']:.3f}",
        ]
        if not single_model:
            cols.append(str(int(row["model count"])))
        rows += " & ".join(cols) + " \\\\\n"

    tabular_spec = "llrccccr" if single_model else "llrccccrc"

    latex = (
        "\\begin{table}[H]\n"
        "\\caption{Caption}\n"
        "\\centering\n"
        "\\resizebox{\\textwidth}{!}{\n"
        f"\\begin{{tabular}}{{{tabular_spec}}}\n" + header + rows + "\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        "\\label{tab:fragile_classes_ab}\n"
        "\\end{table}\n"
    )

    return latex


# def granular_sweep_to_latex(
#     df: pd.DataFrame,
#     model: str,
#     definition_name: str,
#     path: str = "fragile/tables/granular",
# ) -> None:
#     pivot = df.pivot(index="corruption", columns="severity", values="fragile_count")
#     severities = sorted(pivot.columns.tolist())

#     col_spec = "l" + "r" * len(severities)
#     sev_header = " & ".join(f"Sev {s}" for s in severities)
#     header = (
#         "\\toprule\n"
#         f"Corruption & {sev_header} \\\\\n"
#         "\\midrule\n"
#     )

#     groups = (
#         df[["group", "corruption"]]
#         .drop_duplicates()
#         .groupby("group")["corruption"]
#         .apply(list)
#     )
#     rows = ""
#     for group, corruptions in groups.items():
#         rows += f"\\multicolumn{{{len(severities) + 1}}}{{l}}{{\\textit{{{group}}}}} \\\\\n"
#         for corruption in corruptions:
#             if corruption not in pivot.index:
#                 continue
#             vals = " & ".join(
#                 str(int(pivot.loc[corruption, s])) if pd.notna(pivot.loc[corruption, s]) else "--"
#                 for s in severities
#             )
#             rows += f"{corruption.replace('_', chr(92) + '_')} & {vals} \\\\\n"
#         rows += "\\midrule\n"

#     latex = (
#         "\\begin{table}[H]\n"
#         f"\\caption{{Fragile class counts per corruption and severity — {model} ({definition_name})}}\n"
#         "\\centering\n"
#         f"\\begin{{tabular}}{{{col_spec}}}\n"
#         + header
#         + rows
#         + "\\bottomrule\n"
#         "\\end{tabular}\n"
#         f"\\label{{tab:granular_sweep_{model}_{definition_name}}}\n"
#         "\\end{table}\n"
#     )

#     out = Path(path) / model
#     out.mkdir(parents=True, exist_ok=True)
#     filename = f"granular_sweep_{definition_name}.txt"
#     (out / filename).write_text(latex)
#     print(f"Saved to {out / filename}")

def experiment_sweep_to_latex(
    df: pd.DataFrame,
    model: str,
    definition_name: str,
    path: str = "fragile/tables",
) -> None:
    col_spec = "lr"
    header = (
        "\\toprule\n"
        "Experiment & Count \\\\\n"
        "\\midrule\n"
    )
    rows = ""
    for _, row in df.iterrows():
        exp_label = row["experiment"].replace("_", "\\_")
        count = str(int(row["fragile_count"])) if pd.notna(row["fragile_count"]) else "--"
        rows += f"\\hspace{{1em}}{exp_label} & {count} \\\\\n"

    latex = (
        "\\begin{table}[H]\n"
        f"\\caption{{Fragile class counts per experiment — {model} ({definition_name})}}\n"
        "\\centering\n"
        "\\begin{small}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        + header
        + rows
        + "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{small}\n"
        f"\\label{{tab:experiment_sweep_{model}_{definition_name}}}\n"
        "\\end{table}\n"
    )

    out = Path(path) / model
    out.mkdir(parents=True, exist_ok=True)
    filename = f"experiment_sweep_{definition_name}.txt"
    (out / filename).write_text(latex)
    print(f"Saved to {out / filename}")


def granular_sweep_to_latex(
    df: pd.DataFrame,
    model: str,
    definition_name: str,
    path: str = "fragile/tables/granular",
) -> None:
    pivot = df.pivot(index="corruption", columns="severity", values="fragile_count")
    severities = sorted(pivot.columns.tolist())
    col_spec = "l" + "r" * len(severities)
    sev_header = " & ".join(f"Sev {s}" for s in severities)
    header = (
        "\\toprule\n"
        f"Corruption & {sev_header} \\\\\n"
        "\\midrule\n"
    )
    groups = (
        df[["group", "corruption"]]
        .drop_duplicates()
        .groupby("group")["corruption"]
        .apply(list)
    )
    rows = ""
    for group, corruptions in groups.items():
        group_label = group.capitalize()
        rows += (
            f"\\midrule\n"
            f"\\multicolumn{{{len(severities) + 1}}}{{l}}{{"
            f"\\cellcolor{{gray!15}}\\textbf{{{group_label}}}}} \\\\\n"
        )
        for corruption in corruptions:
            if corruption not in pivot.index:
                continue
            vals = " & ".join(
                str(int(pivot.loc[corruption, s])) if pd.notna(pivot.loc[corruption, s]) else "--"
                for s in severities
            )
            corruption_label = corruption.replace("_", "\\_").replace("-", " ").title()
            rows += f"\\hspace{{1em}}{corruption_label} & {vals} \\\\\n"

    latex = (
        "\\begin{table}[H]\n"
        f"\\caption{{Fragile class counts per corruption and severity — {model} ({definition_name})}}\n"
        "\\centering\n"
        "\\begin{small}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        + header
        + rows
        + "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{small}\n"
        f"\\label{{tab:granular_sweep_{model}_{definition_name}}}\n"
        "\\end{table}\n"
    )
    out = Path(path) / model
    out.mkdir(parents=True, exist_ok=True)
    filename = f"granular_sweep_{definition_name}.txt"
    (out / filename).write_text(latex)
    print(f"Saved to {out / filename}")


_CROSS_MODEL_MODELS = {
    "resnet50": "ResNet-50",
    "efficientnet_b4": "EfficientNet-B4",
    "vit_b_16": "ViT-B/16",
    "convnext_base": "ConvNeXt-Base",
}

_CROSS_MODEL_GROUPS = ["blur", "noise", "digital", "weather"]


def generate_fragile_classes_table(
    data: dict[str, dict[str, set]],
    model_labels: dict[str, str] | None = None,
    groups: list[str] | None = None,
    path: str = "fragile/granual",
    filename: str = "cross_model.txt",
    save: bool = False,
) -> str:
    ml = model_labels or _CROSS_MODEL_MODELS
    grp = groups or _CROSS_MODEL_GROUPS

    # Keep only groups that have at least one synset in any model
    grp = [g for g in grp if any(data.get(mk, {}).get(g) for mk in ml)]
    if not grp:
        return ""

    n_groups = len(grp)

    synset_to_label = get_synset_to_label_imagenet1k()

    all_synsets: set[str] = set()
    for model_sets in data.values():
        for s in model_sets.values():
            all_synsets |= s

    def _sort_key(synset: str) -> tuple:
        count = sum(
            1 for mk in ml for g in grp
            if synset in data.get(mk, {}).get(g, set())
        )
        return (-count, synset)

    sorted_synsets = sorted(all_synsets, key=_sort_key)

    col_spec = "ll " + " ".join(["c" * n_groups] * len(ml))

    col_start = 3
    model_headers, cmidrules = [], []
    for model_name in ml.values():
        model_headers.append(f"\\multicolumn{{{n_groups}}}{{c}}{{{model_name}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col_start}-{col_start + n_groups - 1}}}")
        col_start += n_groups

    header1 = "& & " + " & ".join(model_headers) + " \\\\\n"
    cmidrule_line = " ".join(cmidrules) + "\n"

    group_cols = " & ".join(g.capitalize() for g in grp)
    header2 = "Synset & Label & " + " & ".join([group_cols] * len(ml)) + " \\\\\n"

    rows = ""
    for synset in sorted_synsets:
        label = _map_synset(synset, synset_to_label) if synset in synset_to_label else synset
        cells = [
            "\\cellcolor{fragilecolor}\\checkmark"
            if synset in data.get(mk, {}).get(g, set())
            else ""
            for mk in ml
            for g in grp
        ]
        rows += f"{synset} & {label} & " + " & ".join(cells) + " \\\\\n"

    latex = (
        "\\begin{table}[H]\n"
        "\\caption{Fragile classes identified per model and corruption group. "
        "A checkmark and highlighted cell indicate that the class satisfied both Criterion A and Criterion B "
        "across all corruption types and severity levels within the given group.}\n"
        "\\centering\n"
        "\\renewcommand{\\arraystretch}{1.5}\n\n"
        "\\resizebox{\\textwidth}{!}{\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        + header1
        + cmidrule_line
        + header2
        + "\\midrule\n"
        + rows
        + "\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        "\\label{tab:fragile_classes_cross_model}\n"
        "\\end{table}\n"
    )

    if save:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / filename).write_text(latex)
        print(f"Saved to {out / filename}")

    return latex


def cluster_stats_to_latex(
    df: pd.DataFrame,
    save: bool = False,
    filename: str = "cluster_stats.tex",
    path="fragile/tables/stats",
):
    Path(path).mkdir(parents=True, exist_ok=True)
    output = Path(path) / filename
    df.to_latex(output)
