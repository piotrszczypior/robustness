from __future__ import annotations
from pathlib import Path
import pandas as pd

METRICS_PATH = Path("results/representations/resnet50_class_metrics.parquet")
OUT_PATH = Path("results/representations/metrics_table.tex")
MODEL = "resnet50"
N_PER_CLASS = 50

METRIC_COLUMNS = [
    ("angular_distance_median", "Cosine Distance"),
    ("relative_shift_median", "Relative Shift"),
    ("coherence", "Coherence"),
]
FAMILY_ORDER = ["blur", "digital", "noise", "weather"]


def display_corruption(name):
    return name.replace("_", " ").title()


def display_family(name):
    return name.title()


def build_cells(df):
    out = {}
    for src, _ in METRIC_COLUMNS:
        sub = df[df["metric"] == src]
        out[src] = sub.groupby(["corruption", "severity"])["value"].mean().to_dict()
    return out


def ordered_groups(df):
    fam = df.drop_duplicates("corruption").set_index("corruption")["group"].to_dict()
    def fam_key(g):
        lg = g.lower()
        return (FAMILY_ORDER.index(lg) if lg in FAMILY_ORDER else len(FAMILY_ORDER), g)
    fams = sorted(set(fam.values()), key=fam_key)
    return [(f, sorted(c for c in fam if fam[c] == f)) for f in fams]


def fmt(v):
    return f"{v:.2f}" if pd.notna(v) else "--"


def main():
    df = pd.read_parquet(METRICS_PATH)
    if "model" in df.columns:
        df = df[df["model"] == MODEL]
    severities = sorted(int(s) for s in df["severity"].unique())
    cells = build_cells(df)
    groups = ordered_groups(df)

    n_metrics = len(METRIC_COLUMNS)
    n_sev = len(severities)
    total_cols = 1 + n_metrics * n_sev

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\begin{small}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\begin{tabular}{l" + ("".join(" " + "r" * n_sev for _ in METRIC_COLUMNS)) + "}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join(rf"\multicolumn{{{n_sev}}}{{c}}{{{disp}}}" for _, disp in METRIC_COLUMNS) + r" \\")
    lines.append("".join(rf"\cmidrule(lr){{{2 + i * n_sev}-{1 + (i + 1) * n_sev}}}" for i in range(n_metrics)))
    sev_header = " & ".join(f"S{s}" for s in severities)
    lines.append("Corruption & " + " & ".join([sev_header] * n_metrics) + r" \\")
    lines.append(r"\midrule")

    for fam, members in groups:
        lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\cellcolor{{gray!15}}\textbf{{{display_family(fam)}}}}} \\")
        for corr in members:
            vals = []
            for src, _ in METRIC_COLUMNS:
                for s in severities:
                    vals.append(fmt(cells[src].get((corr, s))))
            lines.append(r"\hspace{1em}" + display_corruption(corr) + " & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{small}")
    caption = (
        r"Representation shift for " + MODEL + r" per corruption and severity (S1--S5). "
        r"Each cell is the median across the 1000 ImageNet classes of the per-class median over "
        + str(N_PER_CLASS) + r" images. "
        r"Cosine Distance $=1-\cos(f_{\mathrm{clean}},f_{\mathrm{corr}})$ (rotation), "
        r"Relative Shift $=\lVert\Delta\rVert/\lVert f_{\mathrm{clean}}\rVert$ (magnitude), "
        r"Coherence $=\lVert\frac{1}{n}\sum_i\hat{\Delta}_i\rVert$ (within-class directional consistency, "
        r"chance $\approx1/\sqrt{" + str(N_PER_CLASS) + r"}\approx0.14$)."
    )
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{tab:metric_sweep_" + MODEL + "}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()