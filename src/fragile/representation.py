import re
from utils import get_index_to_synset_and_label_imagenet1k, get_synset_to_label_imagenet1k
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


def arch_contrast_to_latex(
    vit_df: "pd.DataFrame",
    cnn_df: "pd.DataFrame",
    label: str,
    definition_name: str,
    path: str = "fragile/arch_contrast",
    filename: str | None = None,
    save: bool = False,
) -> str:
    synset_to_label = get_synset_to_label_imagenet1k()
    caption_label = label.replace("_", "\\_")

    # col spec: synset, label, 3x ViT, 3x CNN, delta
    col_spec = "ll ccc ccc r"
    header = (
        "\\toprule\n"
        " & & \\multicolumn{3}{c}{ViT} & \\multicolumn{3}{c}{CNN} & \\\\\n"
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n"
        "Synset & Label & Clean & Corrupt & Drop & Clean & Corrupt & Drop & $\\Delta$ \\\\\n"
        "\\midrule\n"
    )

    def _section_rows(df: "pd.DataFrame", delta_col: str, sort_col: str) -> str:
        if df.empty:
            return "\\multicolumn{9}{c}{\\textit{(none)}} \\\\\n"
        rows = ""
        for _, row in df.sort_values(sort_col, ascending=False).iterrows():
            lbl = _map_synset(row["synset"], synset_to_label)
            rows += (
                f"{row['synset']} & {lbl} & "
                f"{row['acc_clean_vit']:.3f} & {row['acc_corrupt_vit']:.3f} & {row['rel_drop_vit']:.3f} & "
                f"{row['acc_clean_cnn']:.3f} & {row['acc_corrupt_cnn']:.3f} & {row['rel_drop_cnn']:.3f} & "
                f"{row[delta_col]:.3f} \\\\\n"
            )
        return rows

    N_COLS = 9
    body = (
        f"\\multicolumn{{{N_COLS}}}{{l}}{{\\textbf{{ViT-exclusive}}}} \\\\\n"
        "\\midrule\n"
        + _section_rows(vit_df, delta_col="delta_vit", sort_col="rel_drop_vit")
        + "\\midrule\n"
        f"\\multicolumn{{{N_COLS}}}{{l}}{{\\textbf{{CNN-exclusive}}}} \\\\\n"
        "\\midrule\n"
        + _section_rows(cnn_df, delta_col="delta_cnn", sort_col="rel_drop_cnn")
    )

    latex = (
        "\\begin{table}[H]\n"
        f"\\caption{{Architecture-specific fragile synsets — {caption_label}}}\n"
        "\\centering\n"
        "\\resizebox{\\textwidth}{!}{\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        + header
        + body
        + "\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        f"\\label{{tab:arch_contrast}}\n"
        "\\end{table}\n"
    )

    if save:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        fname = filename or f"arch_contrast_{label}.txt"
        (out / fname).write_text(latex)
        print(f"Saved to {out / fname}")

    return latex


def dataset_intersection_to_latex(
    synsets: set[str],
    dataset: str,
    definition_name: str,
    path: str = "fragile/dataset_intersection",
    filename: str | None = None,
    save: bool = False,
) -> str:
    synset_to_label = get_synset_to_label_imagenet1k()
    synset_to_index = {v[0]: k for k, v in get_index_to_synset_and_label_imagenet1k().items()}
    
    caption_dataset = dataset.replace("_", "\\_")
    
    header = (
        "\\toprule\n"
        "Synset & Index & Label \\\\\n"
        "\\midrule\n"
    )
    
    rows = ""
    for synset in sorted(synsets):
        label = synset_to_label.get(synset, "Unknown")
        index = synset_to_index.get(synset)
        
        index_str = str(int(index)) if index is not None else "N/A"
        
        rows += f"{synset} & {index_str} & {label} \\\\\n"

    latex = (
        "\\begin{table}[H]\n"
        f"\\caption{{Fragile synsets common to all models — {caption_dataset} ({definition_name})}}\n"
        "\\centering\n"
        "\\begin{tabular}{llr}\n"
        f"{header}"
        f"{rows}"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\label{{tab:dataset_intersection_{definition_name}}}\n"
        "\\end{table}\n"
    )

    if save:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        fname = filename or f"dataset_intersection_{dataset}_{definition_name}.txt"
        file_path = out / fname
        file_path.write_text(latex, encoding="utf-8")
        print(f"Saved to {file_path}")

    return latex


def _pareto_staircase(
    pts: pd.DataFrame, x_col: str, y_col: str, minimize_x: bool
) -> tuple[list[float], list[float]]:
    """Return (xs, ys) staircase coordinates for a 2-D Pareto front.

    minimize_x=True  → ViT front: minimize x, maximize y  (upper-left staircase)
    minimize_x=False → CNN front: maximize x, minimize y  (lower-right staircase)
    """
    if pts.empty:
        return [], []
    sorted_pts = pts.sort_values(x_col, ascending=True)
    xs = sorted_pts[x_col].tolist()
    ys = sorted_pts[y_col].tolist()
    stair_x: list[float] = []
    stair_y: list[float] = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i == 0:
            stair_x.append(x)
            stair_y.append(y)
        else:
            stair_x.append(x)
            stair_y.append(stair_y[-1])
            stair_x.append(x)
            stair_y.append(y)
    return stair_x, stair_y


def arch_contrast_scatter(
    df: pd.DataFrame,
    vit_df: pd.DataFrame,
    cnn_df: pd.DataFrame,
    label: str,
    vit_keys: list[str] | None = None,
    cnn_keys: list[str] | None = None,
    output_dir: str = "fragile/arch_contrast",
) -> None:
    """Scatter plot: x=rel_drop_cnn, y=rel_drop_vit. Pareto points highlighted."""
    import matplotlib.pyplot as plt
    from model import MODELS

    vit_label = MODELS[vit_keys[0]] if vit_keys and len(vit_keys) == 1 else "ViT (average)"
    cnn_label = MODELS[cnn_keys[0]] if cnn_keys and len(cnn_keys) == 1 else "CNN (average)"
    title = f""

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        df["rel_drop_cnn"], df["rel_drop_vit"],
        color="#cccccc", s=18, zorder=1, label="All synsets",
    )

    if not vit_df.empty:
        ax.scatter(
            vit_df["rel_drop_cnn"], vit_df["rel_drop_vit"],
            color="#c0392b", s=20, zorder=3, label=f"{vit_label}-exclusive",
        )
        sx, sy = _pareto_staircase(vit_df, "rel_drop_cnn", "rel_drop_vit", minimize_x=True)
        ax.plot(sx, sy, color="#c0392b", linewidth=1.2, alpha=0.7, zorder=2)

    if not cnn_df.empty:
        ax.scatter(
            cnn_df["rel_drop_cnn"], cnn_df["rel_drop_vit"],
            color="#2563c7", s=20, zorder=3, label=f"{cnn_label}-exclusive",
        )
        sx, sy = _pareto_staircase(cnn_df, "rel_drop_cnn", "rel_drop_vit", minimize_x=False)
        ax.plot(sx, sy, color="#2563c7", linewidth=1.2, alpha=0.7, zorder=2)

    all_x = df["rel_drop_cnn"].tolist()
    all_y = df["rel_drop_vit"].tolist()
    lo = min(min(all_x), min(all_y), -0.05)
    hi = max(max(all_x), max(all_y), 1.0)
    ax.plot([lo, hi], [lo, hi], color="#000000", linestyle="--", linewidth=0.9, alpha=0.8, zorder=2)

    ax.text(0.64, 0.26, f"{cnn_label} more fragile", transform=ax.transAxes,
            fontsize=9, color="#888888", ha="center", va="center", zorder=10)
    ax.text(0.26, 0.74, f"{vit_label} more fragile", transform=ax.transAxes,
            fontsize=9, color="#888888", ha="center", va="center", zorder=10)

    ax.set_xlabel(f"Relative drop {cnn_label}", fontsize=11)
    ax.set_ylabel(f"Relative drop {vit_label}", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    vit_slug = re.sub(r"[^a-z0-9]+", "_", vit_label.lower()).strip("_")
    cnn_slug = re.sub(r"[^a-z0-9]+", "_", cnn_label.lower()).strip("_")
    path = out / f"scatter_{safe_label}_{vit_slug}_vs_{cnn_slug}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter saved to {path}")


def cluster_stats_to_latex(
    df: pd.DataFrame,
    save: bool = False,
    filename: str = "cluster_stats.tex",
    path="fragile/tables/stats",
):
    Path(path).mkdir(parents=True, exist_ok=True)
    output = Path(path) / filename
    df.to_latex(output)
