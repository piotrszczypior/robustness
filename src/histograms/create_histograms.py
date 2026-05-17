import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys


DATA_DIR = Path("results")

CLEAN_CSV = "resnet50_imagenet.csv"
CORRUPT_CSV = "resnet50_imagenet_c_blur_defocus_blur_1.csv"

OUTPUT_DIR = Path("images/histograms")
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD_LOW = 0.25
THRESHOLD_MID = 0.375
THRESHOLD_HIGH = 0.50

COLOR_NEG = "#888780"
COLOR_LOW = "#378ADD"
COLOR_MID = "#E24B4A"
COLOR_HIGH = "#A32D2D"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.2,
        "grid.color": "#888780",
        "figure.dpi": 150,
    }
)


def compute_rel_drop(clean_csv: str, corrupt_csv: str) -> pd.Series:
    clean = pd.read_csv(clean_csv)
    corrupt = pd.read_csv(corrupt_csv)
    acc_clean = clean.groupby("synset")["is_correct"].mean()
    acc_corrupt = corrupt.groupby("synset")["is_correct"].mean()
    df = pd.DataFrame({"clean": acc_clean, "corrupt": acc_corrupt}).dropna()
    rel_drop = (df["clean"] - df["corrupt"]) / df["clean"].replace(0, np.nan)
    return rel_drop.dropna()


def bin_color(mid: float) -> str:
    if mid < 0:
        return COLOR_NEG
    elif mid < THRESHOLD_LOW:
        return COLOR_LOW
    elif mid < THRESHOLD_MID:
        return COLOR_LOW
    elif mid < THRESHOLD_HIGH:
        return COLOR_MID
    else:
        return COLOR_HIGH


def plot_histogram(
    rel_drop: pd.Series, title: str, output_path: Path, n_bins: int = 40
):
    counts, edges = np.histogram(rel_drop, bins=n_bins)
    mids = (edges[:-1] + edges[1:]) / 2
    colors = [bin_color(m) for m in mids]

    n_total = len(rel_drop)
    n_mid = (rel_drop >= THRESHOLD_LOW).sum()
    n_high = (rel_drop >= THRESHOLD_MID).sum()
    n_very_high = (rel_drop >= THRESHOLD_HIGH).sum()

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_width = edges[1] - edges[0]
    ax.bar(mids, counts, width=bar_width * 0.95, color=colors, linewidth=0)

    ax.axvline(THRESHOLD_MID, color=COLOR_MID, linewidth=1.2, linestyle="--", alpha=0.8)
    ax.axvline(
        THRESHOLD_HIGH, color=COLOR_HIGH, linewidth=1.2, linestyle="--", alpha=0.8
    )

    ax.set_xlabel("relative accuracy drop", labelpad=8)
    ax.set_ylabel("number of classes", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="normal", pad=12)

    stats_text = (
        f"n={n_total}  |  ≥25%: {n_mid}  |  ≥37.5%: {n_high}  |  ≥50%: {n_very_high}"
    )
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        color="#888780",
        ha="right",
        va="top",
    )

    patches = [
        mpatches.Patch(color=COLOR_LOW, label="drop < 37.5%"),
        mpatches.Patch(color=COLOR_MID, label="37.5% ≤ drop < 50%"),
        mpatches.Patch(color=COLOR_HIGH, label="drop ≥ 50%"),
        mpatches.Patch(color=COLOR_NEG, label="negative drop"),
    ]
    ax.legend(handles=patches, fontsize=9, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    clean_csv = sys.argv[1] if len(sys.argv) > 1 else CLEAN_CSV
    corrupt_csv = sys.argv[2] if len(sys.argv) > 2 else CORRUPT_CSV
    label = sys.argv[3] if len(sys.argv) > 3 else Path(corrupt_csv).stem

    rel_drop = compute_rel_drop(clean_csv, corrupt_csv)
    out = OUTPUT_DIR / f"hist_{label}.png"
    plot_histogram(rel_drop, title=f"relative accuracy drop — {label}", output_path=out)
