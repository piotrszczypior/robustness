"""Architecture-contrast fragile synset selection.

Independent module — no imports from the rest of the fragile package.

Public API
----------
select_arch_fragile(df, theta_a, theta_min) -> (vit_fragile, cnn_fragile)
plot_arch_contrast_scatter(df, vit_df, cnn_df, ...) -> None

Expected input DataFrame columns
---------------------------------
synset, acc_vit_clean, acc_vit_corrupt, acc_cnn_clean, acc_cnn_corrupt
Optional: label, y_true
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _relative_drop(acc_clean: pd.Series, acc_corrupt: pd.Series) -> pd.Series:
    with np.errstate(invalid="ignore", divide="ignore"):
        drop = (acc_clean - acc_corrupt) / acc_clean
    return drop.where(acc_clean > 0, other=np.nan)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add d_vit, d_cnn, delta_g, asymmetry_vit, asymmetry_cnn to *df* and return a copy."""
    out = df.copy()
    out["d_vit"] = _relative_drop(out["acc_vit_clean"], out["acc_vit_corrupt"])
    out["d_cnn"] = _relative_drop(out["acc_cnn_clean"], out["acc_cnn_corrupt"])
    out["g_clean"] = out["acc_vit_clean"] - out["acc_cnn_clean"]
    out["g_corrupt"] = out["acc_vit_corrupt"] - out["acc_cnn_corrupt"]
    out["delta_g"] = out["g_corrupt"] - out["g_clean"]

    mean_drop = (out["d_vit"] + out["d_cnn"]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        out["asymmetry_vit"] = (out["d_vit"] - out["d_cnn"]) / mean_drop
        out["asymmetry_cnn"] = (out["d_cnn"] - out["d_vit"]) / mean_drop
    out["asymmetry_vit"] = out["asymmetry_vit"].where(mean_drop > 0, other=np.nan)
    out["asymmetry_cnn"] = out["asymmetry_cnn"].where(mean_drop > 0, other=np.nan)

    return out


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def _pareto_indices(
    df: pd.DataFrame, maximize_col: str, minimize_col: str
) -> pd.Index:
    """Return index of non-dominated rows (maximize `maximize_col`, minimize `minimize_col`)."""
    hi = df[maximize_col].values
    lo = df[minimize_col].values
    n = len(hi)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if hi[j] >= hi[i] and lo[j] <= lo[i] and (hi[j] > hi[i] or lo[j] < lo[i]):
                dominated[i] = True
                break
    return df.index[~dominated]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_arch_fragile(
    df: pd.DataFrame,
    theta_a: float = 0.3,
    theta_min: float = 0.1,
    apply_pareto: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select ViT-exclusive and CNN-exclusive fragile synsets.

    Parameters
    ----------
    df:
        DataFrame with columns synset, acc_vit_clean, acc_vit_corrupt,
        acc_cnn_clean, acc_cnn_corrupt.
    theta_a:
        Minimum normalised asymmetry score required.
    theta_min:
        Minimum absolute relative drop for the dominant architecture.
    apply_pareto:
        If True, apply a secondary Pareto filter on top of threshold selection.

    Returns
    -------
    (vit_fragile, cnn_fragile, excluded_negative_drop)
    delta_g is kept in output for reference but not used in selection.
    Synsets where d_vit <= 0 or d_cnn <= 0 are excluded before asymmetry
    computation and returned separately in excluded_negative_drop.
    """
    enriched = compute_metrics(df).dropna(subset=["d_vit", "d_cnn", "asymmetry_vit", "asymmetry_cnn"])

    positive_drop = (enriched["d_vit"] > 0) & (enriched["d_cnn"] > 0)
    excluded_negative_drop = enriched[~positive_drop].copy()
    enriched = enriched[positive_drop]

    vit_mask = (enriched["asymmetry_vit"] > theta_a) & (enriched["d_vit"] > theta_min)
    cnn_mask = (enriched["asymmetry_cnn"] > theta_a) & (enriched["d_cnn"] > theta_min)

    vit_candidates = enriched[vit_mask].copy()
    cnn_candidates = enriched[cnn_mask].copy()

    if apply_pareto and not vit_candidates.empty:
        idx = _pareto_indices(vit_candidates, maximize_col="d_vit", minimize_col="d_cnn")
        vit_candidates = vit_candidates.loc[idx]

    if apply_pareto and not cnn_candidates.empty:
        idx = _pareto_indices(cnn_candidates, maximize_col="d_cnn", minimize_col="d_vit")
        cnn_candidates = cnn_candidates.loc[idx]

    # remove synsets that ended up in both sets
    overlap = set(vit_candidates["synset"]) & set(cnn_candidates["synset"])
    if overlap:
        vit_candidates = vit_candidates[~vit_candidates["synset"].isin(overlap)]
        cnn_candidates = cnn_candidates[~cnn_candidates["synset"].isin(overlap)]

    return vit_candidates, cnn_candidates, excluded_negative_drop


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def _pareto_staircase(
    pts: pd.DataFrame, x_col: str, y_col: str
) -> tuple[list[float], list[float]]:
    if pts.empty:
        return [], []
    sorted_pts = pts.sort_values(x_col, ascending=True)
    xs = sorted_pts[x_col].tolist()
    ys = sorted_pts[y_col].tolist()
    sx: list[float] = []
    sy: list[float] = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i == 0:
            sx.append(x)
            sy.append(y)
        else:
            sx.append(x)
            sy.append(sy[-1])
            sx.append(x)
            sy.append(y)
    return sx, sy


def plot_arch_contrast_scatter(
    df: pd.DataFrame,
    vit_df: pd.DataFrame,
    cnn_df: pd.DataFrame,
    vit_label: str = "ViT",
    cnn_label: str = "CNN",
    title: str = "",
    output_path: str | None = None,
    output_dir: str = "fragile/arch_contrast",
) -> None:
    """Scatter plot: x=d_cnn, y=d_vit with Pareto step-fronts.

    Parameters
    ----------
    df:         Full enriched DataFrame (output of compute_metrics).
    vit_df:     ViT-exclusive Pareto synsets.
    cnn_df:     CNN-exclusive Pareto synsets.
    vit_label:  Display name for the ViT architecture.
    cnn_label:  Display name for the CNN architecture.
    output_path: Explicit save path (overrides output_dir + auto-naming).
    output_dir:  Directory for auto-named file.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))

    all_x = df["d_cnn"].dropna().tolist()
    all_y = df["d_vit"].dropna().tolist()
    lo = min(min(all_x), min(all_y), -0.05)
    hi = max(max(all_x), max(all_y), 1.0)

    ax.scatter(df["d_cnn"], df["d_vit"], color="#cccccc", s=18, zorder=1, label="All synsets")

    if not vit_df.empty:
        ax.scatter(
            vit_df["d_cnn"], vit_df["d_vit"],
            color="#c0392b", s=22, zorder=3, label=f"{vit_label} (Pareto)",
        )
        sx, sy = _pareto_staircase(vit_df, "d_cnn", "d_vit")
        ax.plot(sx, sy, color="#c0392b", linewidth=1.2, alpha=0.7, zorder=2)

    if not cnn_df.empty:
        ax.scatter(
            cnn_df["d_cnn"], cnn_df["d_vit"],
            color="#2563c7", s=22, zorder=3, label=f"{cnn_label} (Pareto)",
        )
        sx, sy = _pareto_staircase(cnn_df, "d_cnn", "d_vit")
        ax.plot(sx, sy, color="#2563c7", linewidth=1.2, alpha=0.7, zorder=2)

    ax.plot([lo, hi], [lo, hi], color="#000000", linestyle="--", linewidth=0.9, alpha=0.8, zorder=2)

    ax.text(0.74, 0.26, f"{cnn_label} more fragile", transform=ax.transAxes,
            fontsize=9, color="#888888", ha="center", va="center", zorder=10)
    ax.text(0.26, 0.74, f"{vit_label} more fragile", transform=ax.transAxes,
            fontsize=9, color="#888888", ha="center", va="center", zorder=10)

    ax.set_xlabel(f"Relative drop — {cnn_label}", fontsize=11)
    ax.set_ylabel(f"Relative drop — {vit_label}", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    if output_path is None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        vit_slug = re.sub(r"[^a-z0-9]+", "_", vit_label.lower()).strip("_")
        cnn_slug = re.sub(r"[^a-z0-9]+", "_", cnn_label.lower()).strip("_")
        label_slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "scatter"
        output_path = str(out / f"{label_slug}_{vit_slug}_vs_{cnn_slug}.png")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter saved to {output_path}")
