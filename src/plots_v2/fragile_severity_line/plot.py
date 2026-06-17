from pathlib import Path
import matplotlib.pyplot as plt

SEVERITIES = [1, 2, 3, 4, 5]

_COLORS  = ["#4D99CB", "#C54C3F", "#5BAD6F", "#9B59B6"]
_MARKERS = ["o", "s", "^", "D"]


def render(data: dict[str, list[int]], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")

    for (model_label, counts), color, marker in zip(data.items(), _COLORS, _MARKERS):
        ax.plot(
            SEVERITIES, counts,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=1.5,
            label=model_label,
        )

    ax.set_xlabel("Severity", fontsize=11)
    ax.set_ylabel("Fragile class count", fontsize=11)
    ax.set_title(title, fontsize=13)

    ax.set_xticks(SEVERITIES)
    ax.tick_params(axis="both", labelsize=10)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=9, frameon=True, framealpha=0.95, edgecolor="#dddddd")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from utils import save_as_pdf
    save_as_pdf(fig, output_path)
    plt.close(fig)