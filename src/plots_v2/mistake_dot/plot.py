from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

_COLOR = "#3B6FD4"
_ALPHA = 0.85
_LW = 5.0


def prepare_synset_data(
    records: list[dict],
    synset: str,
    index_to_label: dict[int, list[str]],
    synset_to_index: dict[str, int],
    top_k: int,
    min_count: int,
) -> list[tuple[str, int]]:
    """Returns [(pred_label, count), ...] sorted by count desc, top_k entries."""
    true_idx = synset_to_index.get(synset)
    counts: dict[str, int] = defaultdict(int)

    for r in records:
        if r["synset"] != synset:
            continue
        y_pred = int(r["y_pred"])
        count = int(r["count"])
        if true_idx is not None and y_pred == true_idx:
            continue
        if count < min_count:
            continue
        entry = index_to_label.get(y_pred, [str(y_pred), str(y_pred)])
        pred_label = entry[1].replace("_", " ")
        counts[pred_label] += count

    sorted_preds = sorted(counts.items(), key=lambda x: -x[1])
    return sorted_preds[:top_k]


def render(
    entries: list[tuple[str, int]],
    synset: str,
    synset_label: str,
    model_label: str,
    task_name: str,
    output_path: Path,
) -> None:
    if not entries:
        print(f"  No data for {synset}, skipping.")
        return

    y_labels = [label for label, _ in entries]
    counts = [count for _, count in entries]
    n_y = len(entries)
    x_max = max(counts)

    fig_h = max(3, n_y * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(max(8, x_max * 0.6 + 3), fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, count in enumerate(counts):
        ax.hlines(i, 0, count, colors=_COLOR, linewidth=_LW, alpha=_ALPHA, zorder=3)

    # Full vertical grid at every integer x
    for x in range(1, x_max + 1):
        ax.axvline(x, color="#e8e8e8", linewidth=0.6, zorder=1)

    # Horizontal separators between rows
    for i in range(n_y - 1):
        ax.axhline(i + 0.5, color="#cccccc", linewidth=0.6, zorder=2)

    ax.set_yticks(range(n_y))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_ylim(-0.6, n_y - 0.4)
    ax.invert_yaxis()

    ax.set_xlabel("Count", fontsize=10)
    ax.set_xlim(0, x_max + 0.5)
    ax.set_xticks(range(0, x_max + 1))

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["left"].set_color("#aaaaaa")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("#aaaaaa")

    fig.suptitle(
        f"{synset_label} — {model_label} — {task_name}",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
