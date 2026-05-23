from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

SEVERITIES = [1, 2, 3, 4, 5]

_COLORS      = ["#3B6FD4", "#D4503B", "#3BAD6F", "#9B59B6"]
_MARKERSIZES = [13, 10, 7, 4]


def render(data: dict[str, list[int]], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for (model_label, counts), color, msize in zip(data.items(), _COLORS, _MARKERSIZES):
        ax.plot(
            SEVERITIES, counts,
            color=color,
            marker="o",
            markersize=msize,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=2,
            linewidth=1.5,
            solid_capstyle="round",
            label=model_label,
        )

    ax.set_xlabel("Severity", fontsize=11, color="#555555", labelpad=8)
    ax.set_ylabel("Fragile class count", fontsize=11, color="#555555", labelpad=8)
    ax.set_title(title, fontsize=13, pad=10, color="#1a1a1a")

    ax.set_xticks(SEVERITIES)
    ax.tick_params(axis="both", labelsize=10, colors="#777777", length=3)

    ax.grid(True, axis="y", which="major", linestyle="-", linewidth=0.4, alpha=0.15, color="#000000")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color("#aaaaaa")

    handles = [
        mlines.Line2D(
            [], [],
            color=color,
            marker="o",
            markersize=msize,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=2,
            linewidth=1.5,
            label=model_label,
        )
        for (model_label, _), color, msize in zip(data.items(), _COLORS, _MARKERSIZES)
    ]

    ax.legend(
        handles=handles,
        fontsize=9,
        frameon=True,
        framealpha=0.95,
        edgecolor="#dddddd",
        facecolor="white",
        handlelength=2.4,
        handletextpad=0.6,
        labelcolor="#333333",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)