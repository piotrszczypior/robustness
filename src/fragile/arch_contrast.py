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
    out["abs_drop_vit"] = out["acc_vit_clean"] - out["acc_vit_corrupt"]
    out["abs_drop_cnn"] = out["acc_cnn_clean"] - out["acc_cnn_corrupt"]
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

def select_arch_exclusive_ab(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select synsets where A∩B holds for exactly one architecture.

    Red  (ViT-exclusive): A∩B for ViT AND NOT A∩B for CNN.
    Blue (CNN-exclusive): A∩B for CNN AND NOT A∩B for ViT.

    A: acc_clean >= 0.80 AND acc_corrupt <= 0.50
    B: rel_drop >= 75th percentile of own distribution
    """
    enriched = compute_metrics(df).dropna(subset=["d_vit", "d_cnn"])

    p75_vit = np.percentile(enriched["d_vit"].dropna(), 75)
    p75_cnn = np.percentile(enriched["d_cnn"].dropna(), 75)

    ab_vit = (
        (enriched["acc_vit_clean"] >= 0.80)
        & (enriched["acc_vit_corrupt"] <= 0.50)
        & (enriched["d_vit"] >= p75_vit)
    )
    ab_cnn = (
        (enriched["acc_cnn_clean"] >= 0.80)
        & (enriched["acc_cnn_corrupt"] <= 0.50)
        & (enriched["d_cnn"] >= p75_cnn)
    )

    robust_vit = (enriched["acc_vit_clean"] >= 0.80) & (enriched["acc_vit_corrupt"] >= 0.50)
    robust_cnn = (enriched["acc_cnn_clean"] >= 0.80) & (enriched["acc_cnn_corrupt"] >= 0.50)

    vit_excl = enriched[ab_vit & ~ab_cnn & robust_cnn].copy()
    cnn_excl = enriched[ab_cnn & ~ab_vit & robust_vit].copy()

    return vit_excl, cnn_excl


def select_arch_fragile(
    df: pd.DataFrame,
    apply_pareto: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select ViT-exclusive and CNN-exclusive fragile synsets via Pareto filter.

    Parameters
    ----------
    df:
        DataFrame with columns synset, acc_vit_clean, acc_vit_corrupt,
        acc_cnn_clean, acc_cnn_corrupt.
    apply_pareto:
        If True, apply Pareto filter (maximize own drop, minimize other's drop).

    Returns
    -------
    (vit_fragile, cnn_fragile, excluded_negative_drop)
    Synsets where d_vit <= 0 or d_cnn <= 0 are excluded and returned separately.
    """
    enriched = compute_metrics(df).dropna(subset=["d_vit", "d_cnn"])
    excluded_negative_drop = pd.DataFrame()

    p75_vit = np.percentile(enriched["d_vit"].dropna(), 75)
    p75_cnn = np.percentile(enriched["d_cnn"].dropna(), 75)

    ab_vit = (
        (enriched["acc_vit_clean"] >= 0.80)
        & (enriched["acc_vit_corrupt"] <= 0.50)
        &
        (enriched["d_vit"] >= p75_vit)
    )
    ab_cnn = (
        (enriched["acc_cnn_clean"] >= 0.80)
        & (enriched["acc_cnn_corrupt"] <= 0.50)
        &
        (enriched["d_cnn"] >= p75_cnn)
    )
    gap = (enriched["acc_vit_corrupt"] - enriched["acc_cnn_corrupt"]).abs() >= 0.2
    enriched = enriched[(ab_vit | ab_cnn) & gap]

    robust_vit = (enriched["acc_vit_clean"] >= 0.80) & (enriched["acc_vit_corrupt"] >= 0.50)
    robust_cnn = (enriched["acc_cnn_clean"] >= 0.80) & (enriched["acc_cnn_corrupt"] >= 0.50)

    vit_candidates = enriched[(enriched["d_vit"] > enriched["d_cnn"]) & robust_cnn].copy()
    cnn_candidates = enriched[(enriched["d_cnn"] > enriched["d_vit"]) & robust_vit].copy()

    if apply_pareto and not vit_candidates.empty:
        idx = _pareto_indices(vit_candidates, maximize_col="acc_cnn_corrupt", minimize_col="acc_vit_corrupt")
        vit_candidates = vit_candidates.loc[idx]

    if apply_pareto and not cnn_candidates.empty:
        idx = _pareto_indices(cnn_candidates, maximize_col="acc_vit_corrupt", minimize_col="acc_cnn_corrupt")
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


_METRIC_COLS = {
    "rel_drop":   ("d_cnn",          "d_vit",          "Relative drop",    "Relative drop"),
    "abs_drop":   ("abs_drop_cnn",   "abs_drop_vit",   "Absolute drop",    "Absolute drop"),
    "acc_corrupt": ("acc_cnn_corrupt", "acc_vit_corrupt", "Corrupt accuracy", "Corrupt accuracy"),
}


def plot_arch_contrast_scatter(
    df: pd.DataFrame,
    vit_df: pd.DataFrame,
    cnn_df: pd.DataFrame,
    vit_label: str = "ViT",
    cnn_label: str = "CNN",
    title: str = "",
    severity: int | str | None = None,
    metric: str = "rel_drop",
    synset_labels: dict[str, str] | None = None,
    output_path: str | None = None,
    output_dir: str = "fragile/arch_contrast",
) -> None:
    """Scatter plot with Pareto step-fronts for arch-contrast.

    Parameters
    ----------
    df:          Full enriched DataFrame (output of compute_metrics).
    vit_df:      ViT-exclusive Pareto synsets.
    cnn_df:      CNN-exclusive Pareto synsets.
    vit_label:   Display name for the ViT architecture.
    cnn_label:   Display name for the CNN architecture.
    metric:      Axis metric — "rel_drop" | "abs_drop" | "acc_corrupt".
    output_path: Explicit save path (overrides output_dir + auto-naming).
    output_dir:  Directory for auto-named file.
    """
    import matplotlib.pyplot as plt

    if metric not in _METRIC_COLS:
        raise ValueError(f"metric must be one of {list(_METRIC_COLS)}; got {metric!r}")

    x_col, y_col, x_prefix, y_prefix = _METRIC_COLS[metric]

    fig, ax = plt.subplots(figsize=(7, 7))

    all_x = df[x_col].dropna().tolist()
    all_y = df[y_col].dropna().tolist()
    lo = min(min(all_x), min(all_y))
    hi = max(max(all_x), max(all_y))
    pad = (hi - lo) * 0.04
    lo -= pad
    hi += pad

    ax.scatter(df[x_col], df[y_col], color="#cccccc", s=18, zorder=1, label="All synsets")

    def _annotate(subset: pd.DataFrame, color: str) -> None:
        for i, (_, row) in enumerate(subset.iterrows()):
            synset = row.get("synset", "")
            lbl = synset_labels.get(synset, synset) if synset_labels else synset
            print(lbl, i)
            ax.annotate(
                lbl,
                xy=(row[x_col], row[y_col]),
                xytext=(-4, 4) if not (i == 5 or i == 2) else (-4, -4),
                textcoords="offset points",
                ha="right",
                fontsize=7,
                color=color,
                zorder=4,
            )

    if not vit_df.empty:
        ax.scatter(
            vit_df[x_col], vit_df[y_col],
            color="#c0392b", s=22, zorder=3, label=f"Fragile {vit_label}",
        )
        _annotate(vit_df, "#c0392b")

    if not cnn_df.empty:
        ax.scatter(
            cnn_df[x_col], cnn_df[y_col],
            color="#2563c7", s=22, zorder=3, label=f"Fragile {cnn_label}",
        )
        _annotate(cnn_df, "#2563c7")

    ax.plot([lo, hi], [lo, hi], color="#000000", linestyle="--", linewidth=0.9, alpha=0.8, zorder=2)

    ax.set_xlabel(f"{x_prefix} — {cnn_label}", fontsize=11)
    ax.set_ylabel(f"{y_prefix} — {vit_label}", fontsize=11)
    sev_str = f"severity {severity}" if severity is not None else "severity all"
    ax.set_title(f"{title.capitalize()}\n{sev_str}", fontsize=11)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    if output_path is None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        vit_slug = re.sub(r"[^a-z0-9]+", "_", vit_label.lower()).strip("_")
        cnn_slug = re.sub(r"[^a-z0-9]+", "_", cnn_label.lower()).strip("_")
        label_slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "scatter"
        output_path = str(out / f"{label_slug}_{metric}_{vit_slug}_vs_{cnn_slug}.png")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter saved to {output_path}")
