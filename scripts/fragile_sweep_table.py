from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES
from fragile.data import get_per_class_accuracy
from fragile.fragile import get_absolute_fragile, get_relative_drop_fragile
from fragile.methods import calculate_relative_drop
from model import MODELS


# canonical column order matching the LaTeX table
_GROUPS: list[tuple[str, str, list[tuple[str, str]]]] = [
    ("blur", "Blur", [
        ("defocus_blur",    "Defocus Blur"),
        ("glass_blur",      "Glass Blur"),
        ("motion_blur",     "Motion Blur"),
        ("zoom_blur",       "Zoom Blur"),
    ]),
    ("digital", "Digital", [
        ("contrast",           "Contrast"),
        ("elastic_transform",  "Elastic Trans."),
        ("jpeg_compression",   "JPEG Comp."),
        ("pixelate",           "Pixelate"),
    ]),
    ("noise", "Noise", [
        ("gaussian_noise", "Gaussian Noise"),
        ("impulse_noise",  "Impulse Noise"),
        ("shot_noise",     "Shot Noise"),
    ]),
    ("weather", "Weather", [
        ("brightness", "Brightness"),
        ("fog",        "Fog"),
        ("frost",      "Frost"),
        ("snow",       "Snow"),
    ]),
]

_ALL_CORRUPTIONS: list[tuple[str, str, str]] = [
    (group, c, label)
    for group, _, pairs in _GROUPS
    for c, label in pairs
]


def _fragile_count(model: str, group: str, corruption: str, severity: int, data_path: Path) -> int | None:
    try:
        clean = get_per_class_accuracy(f"{model}_imagenet.csv", data_path, agg_column="acc_clean")
        corrupt = get_per_class_accuracy(
            f"{model}_imagenet_c_{group}_{corruption}_{severity}.csv", data_path
        )
        df = clean.merge(corrupt.rename(columns={"accuracy": "acc_corrupt"}), on="synset").dropna()
        if df.empty:
            return None
        df = calculate_relative_drop(df)
        df = get_absolute_fragile(df)
        df = get_relative_drop_fragile(df)
        return int(((df["is_fragile_a"] == 1) & (df["is_fragile_b"] == 1)).sum())
    except FileNotFoundError:
        return None


def build_data(models: list[str], data_path: Path) -> pd.DataFrame:
    """Returns DataFrame indexed by (model, severity), columns = corruption names."""
    records = []
    total = len(models) * len(IMAGENET_C_SEVERITIES) * len(_ALL_CORRUPTIONS)
    done = 0
    for model in models:
        for severity in IMAGENET_C_SEVERITIES:
            row: dict = {"model": model, "severity": severity}
            for group, corruption, _ in _ALL_CORRUPTIONS:
                count = _fragile_count(model, group, corruption, severity, data_path)
                row[corruption] = count
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{total}", file=sys.stderr)
            records.append(row)
    return pd.DataFrame(records).set_index(["model", "severity"])


def _fmt(val: int | None, bold: bool) -> str:
    if val is None:
        return "--"
    s = str(val)
    return rf"\textbf{{{s}}}" if bold else s


def build_latex(df: pd.DataFrame, models: list[str]) -> str:
    corruption_names = [c for _, c, _ in _ALL_CORRUPTIONS]
    n_cols = len(corruption_names)

    # per (corruption, severity): minimum across models
    mins: dict[tuple[str, int], int] = {}
    for corruption in corruption_names:
        for severity in IMAGENET_C_SEVERITIES:
            vals = [df.loc[(m, severity), corruption] for m in models
                    if (m, severity) in df.index and pd.notna(df.loc[(m, severity), corruption])]
            if vals:
                mins[(corruption, severity)] = int(min(vals))

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\caption{Number of fragile classes identified per model and severity level "
                 r"across all corruption types. A class is considered fragile if it satisfies "
                 r"both Criterion~A and Criterion~B simultaneously. Bold values indicate the "
                 r"minimum count for each corruption type and severity level across all "
                 r"architectures. S denotes severity level.}")
    lines.append(r"\centering")
    lines.append(r"\begin{small}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")

    # column spec: ll + r per corruption
    col_spec = "ll " + " ".join(["r"] * n_cols)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # group headers
    group_header = " & & "
    cmidrules = []
    col_idx = 3
    for _, group_label, pairs in _GROUPS:
        n = len(pairs)
        group_header += rf"\multicolumn{{{n}}}{{c}}{{{group_label}}}"
        if (_, group_label, pairs) != _GROUPS[-1]:
            group_header += " & "
        cmidrules.append(rf"\cmidrule(lr){{{col_idx}-{col_idx + n - 1}}}")
        col_idx += n
    lines.append(group_header + r" \\")
    lines.append("".join(cmidrules))

    # rotated column headers
    rot_header = r"Model & S" + "".join(
        rf" & \rotatebox{{90}}{{{label}\;}}"
        for _, _, pairs in _GROUPS
        for _, label in pairs
    ) + r" \\"
    lines.append(rot_header)
    lines.append(r"\midrule")

    for model in models:
        model_label = MODELS.get(model, model)
        for i, severity in enumerate(IMAGENET_C_SEVERITIES):
            if (model, severity) not in df.index:
                continue
            row = df.loc[(model, severity)]
            cells = []
            for corruption in corruption_names:
                val = row[corruption]
                val_int = int(val) if pd.notna(val) else None
                bold = val_int is not None and mins.get((corruption, severity)) == val_int
                cells.append(_fmt(val_int, bold))

            if i == 0:
                model_col = rf"\multirow{{5}}{{*}}{{{model_label}}}"
            else:
                model_col = ""

            lines.append(f" {model_col} & {severity} & " + " & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    # replace last \midrule with \bottomrule
    lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{small}")
    lines.append(r"\label{tab:fragile_sweep_ab}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fragile count sweep table (definition ab)")
    parser.add_argument("--data-path", default="results")
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated model keys (default: all from MODELS)"
    )
    parser.add_argument("--out", default=None, help="Output .tex path (default: stdout)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else list(MODELS.keys())
    )

    print(f"Sweep: {len(models)} models × {len(IMAGENET_C_SEVERITIES)} severities × {len(_ALL_CORRUPTIONS)} corruptions", file=sys.stderr)
    df = build_data(models, data_path)

    tex = build_latex(df, models)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex + "\n")
        print(f"Saved: {out}", file=sys.stderr)
    else:
        print(tex)


if __name__ == "__main__":
    main()
