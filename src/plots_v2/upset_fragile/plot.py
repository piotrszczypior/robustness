from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fragile.fragile import get_absolute_fragile, get_relative_drop_fragile

GROUPS = ["blur", "noise", "weather", "digital"]

_COLOR = "#1f77b4"
_DOT_INACTIVE = "#cccccc"
_DOT_SIZE = 120


def fragile_synsets(df: pd.DataFrame) -> set:
    df = get_absolute_fragile(df)
    df = get_relative_drop_fragile(df)
    mask = (df["is_fragile_a"] == 1) & (df["is_fragile_b"] == 1)
    return set(df.loc[mask, "synset"])


def _intersection_counts(sets: dict[str, set], ordered_keys: list[str]) -> list[tuple[tuple, int]]:
    from itertools import product
    results = []
    for combo in product([False, True], repeat=len(ordered_keys)):
        if not any(combo):
            continue
        active_keys = [k for k, active in zip(ordered_keys, combo) if active]
        intersection = set.intersection(*(sets[k] for k in active_keys))
        if intersection:
            results.append((combo, len(intersection)))
    return sorted(results, key=lambda x: -x[1])


def render(sets: dict[str, set], title: str, output_path: Path, groups: list[str] | None = None) -> None:
    ordered = groups if groups is not None else GROUPS
    n_groups = len(ordered)
    sorted_intersections = _intersection_counts(sets, ordered)
    combos = [c for c, _ in sorted_intersections]
    values = [v for _, v in sorted_intersections]
    n = len(combos)

    _BAR_HEIGHT = 3.75
    _DOT_ROW_HEIGHT = 0.6
    dots_height = n_groups * _DOT_ROW_HEIGHT
    fig_width = max(8, n * 0.9)
    fig, (ax_bars, ax_dots) = plt.subplots(
        2, 1,
        figsize=(fig_width, _BAR_HEIGHT + dots_height),
        gridspec_kw={"height_ratios": [_BAR_HEIGHT, dots_height]},
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.05)

    x = list(range(n))

    ax_bars.bar(x, values, color=_COLOR, width=0.5, zorder=2)
    for i, v in enumerate(values):
        ax_bars.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    ax_bars.set_ylabel("Intersection size", fontsize=12)
    ax_bars.set_xlim(-0.5, n - 0.5)
    ax_bars.set_xticks([])
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)
    ax_bars.spines["bottom"].set_visible(False)
    ax_bars.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax_bars.set_axisbelow(True)
    ax_bars.set_title(title, fontsize=18, pad=8)

    for col_idx, combo in enumerate(combos):
        active_ys = [n_groups - 1 - r for r, active in enumerate(combo) if active]
        if len(active_ys) > 1:
            ax_dots.plot(
                [col_idx, col_idx], [min(active_ys), max(active_ys)],
                color=_COLOR, linewidth=2, zorder=1, solid_capstyle="round",
            )
        for row_idx, active in enumerate(combo):
            y = n_groups - 1 - row_idx
            ax_dots.scatter(
                col_idx, y,
                c=_COLOR if active else _DOT_INACTIVE,
                s=_DOT_SIZE, zorder=2, linewidths=0,
            )

    ax_dots.set_xlim(-0.5, n - 0.5)
    ax_dots.set_ylim(-0.5, n_groups - 0.5)
    ax_dots.set_yticks(list(range(n_groups)))
    ax_dots.set_yticklabels([g.replace("_", " ").capitalize() for g in reversed(ordered)], fontsize=12)
    ax_dots.set_xticks([])
    for spine in ax_dots.spines.values():
        spine.set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
