from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import MODELS

from constants import IMAGENET_C_SEVERITIES


_COLORS = ["black", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#924bd5"]
_LABELS = ["clean", "sev 1", "sev 2", "sev 3", "sev 4", "sev 5"]

_TOP_N = 5
_MID_N = 5
_BOT_N = 5


def select_synsets(
    severity_dfs: dict, top_n=_TOP_N, mid_n=_MID_N, bot_n=_BOT_N
) -> list[str]:
    frames = [df[["synset", "acc_clean"]] for df in severity_dfs[1].values()]
    avg_clean = (
        pd.concat(frames, ignore_index=True)
        .groupby("synset")["acc_clean"]
        .mean()
        .sort_values(ascending=False)
    )
    synsets = avg_clean.index.tolist()
    n = len(synsets)

    top = synsets[:top_n]

    median_idx = (avg_clean - avg_clean.median()).abs().argmin()
    mid_start = max(0, median_idx - mid_n // 2)
    mid = synsets[mid_start : mid_start + mid_n]

    bot = synsets[n - bot_n :]
    return top + mid + bot


def select_synsets_single_model(
    severity_dfs: dict, top_n=_TOP_N, mid_n=_MID_N, bot_n=_BOT_N
) -> list[str]:
    avg_clean = (
        severity_dfs[1]
        .groupby("synset")["acc_clean"]
        .mean()
        .sort_values(ascending=False)
    )
    synsets = avg_clean.index.tolist()
    n = len(synsets)

    top = synsets[:top_n]

    median_idx = (avg_clean - avg_clean.median()).abs().argmin()
    mid_start = max(0, median_idx - mid_n // 2)
    mid = synsets[mid_start : mid_start + mid_n]

    bot = synsets[n - bot_n :]
    return top + mid + bot


def render(
    severity_dfs: dict,
    model_name: str,
    group_name: str,
    output_path: Path,
    selected_synsets: list[str],
) -> None:
    x = np.arange(len(selected_synsets))
    n_plots = 1 + len(IMAGENET_C_SEVERITIES)

    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 3 * n_plots), sharex=True)
    fig.patch.set_facecolor("white")

    top_n = _TOP_N
    mid_n = _MID_N

    clean_series = severity_dfs[1].set_index("synset")["acc_clean"]
    clean_vals = np.array([clean_series.get(s, np.nan) for s in selected_synsets])

    axes[0].scatter(x, clean_vals, c=_COLORS[0], alpha=0.8, s=18)
    axes[0].set_ylabel("Accuracy", fontsize=20)
    axes[0].set_title("Clean", fontsize=20)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    for sev in IMAGENET_C_SEVERITIES:
        ax = axes[sev]
        corrupt_series = severity_dfs[sev].set_index("synset")["acc_corrupt"]
        vals = np.array([corrupt_series.get(s, np.nan) for s in selected_synsets])

        for xi, (c, v) in enumerate(zip(clean_vals, vals)):
            if not np.isnan(c) and not np.isnan(v):
                ax.plot(
                    [xi, xi],
                    [c, v],
                    color="#444444",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.6,
                )

        ax.scatter(x, vals, c=_COLORS[sev], alpha=0.8, s=18)
        ax.scatter(x, clean_vals, c=_COLORS[0], alpha=0.3, s=12)
        ax.set_ylabel("Accuracy", fontsize=20)
        ax.set_title(f"Severity {sev}", fontsize=20)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, linestyle=":", alpha=0.5)

    for ax in axes:
        ax.axvline(top_n - 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.axvline(
            top_n + mid_n - 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.8
        )
        ax.set_xlim(-0.5, len(selected_synsets) - 0.5)

    try:
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(selected_synsets, rotation=90, fontsize=14)
    except Exception:
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(x, fontsize=14)

    axes[-1].set_xlabel("Classes", labelpad=15, fontsize=20)
    fig.suptitle(
        f"Impact of {group_name} corruption severity on per-class accuracy - {MODELS[model_name]}",
        fontsize=24,
        y=1.01,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
