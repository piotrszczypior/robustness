from pathlib import Path

import matplotlib.pyplot as plt

_COLOR_MISTAKE = "#3B6FD4"
_COLOR_CORRECT = "#2E8B57"
_ALPHA = 0.85
_LW = 5.0


def prepare_synset_data(
    records: list[dict],
    synset: str,
    index_to_label: dict[int, list[str]],
    synset_to_index: dict[str, int],
    top_k: int,
    min_count: int,
) -> list[dict]:
    """Returns list of dicts with label, synset, count, is_correct."""
    true_idx = synset_to_index.get(synset)
    all_preds = []

    for r in records:
        if r["synset"] != synset:
            continue
        y_pred = int(r["y_pred"])
        count = int(r["count"])

        is_correct = (true_idx is not None and y_pred == true_idx)

        # Skip mistakes with low count, but always keep correct one if it exists
        if not is_correct and count < min_count:
            continue

        entry = index_to_label.get(y_pred, [str(y_pred), str(y_pred)])
        pred_synset = entry[0]
        pred_label = entry[1].replace("_", " ")

        all_preds.append({
            "label": pred_label,
            "synset": pred_synset,
            "count": count,
            "is_correct": is_correct
        })

    # Sort by count descending
    all_preds.sort(key=lambda x: -x["count"])

    return all_preds[:top_k]


def render(
    entries: list[dict],
    synset: str,
    synset_label: str,
    model_label: str,
    task_name: str,
    output_path: Path,
) -> None:
    if not entries:
        print(f"  No data for {synset}, skipping.")
        return

    y_labels = [f"{e['label']}\n({e['synset']})" for e in entries]
    counts = [e['count'] for e in entries]
    is_correct_list = [e['is_correct'] for e in entries]
    n_y = len(entries)
    x_max = max(counts)
    x_limit = max(50, x_max)

    fig_h = max(3, n_y * 0.7 + 1.5)
    fig, ax = plt.subplots(figsize=(max(8, x_max * 0.1 + 4), fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (count, is_correct) in enumerate(zip(counts, is_correct_list)):
        color = _COLOR_CORRECT if is_correct else _COLOR_MISTAKE
        ax.hlines(i, 0, count, colors=color, linewidth=_LW, alpha=_ALPHA, zorder=3)
        # Add a dot at the end
        ax.scatter(count, i, color=color, s=50, zorder=4)

    # Ticks every 5, and explicitly include 50
    xticks = list(range(0, int(x_limit) + 1, 5))
    if 50 not in xticks:
        xticks.append(50)
    xticks = sorted(list(set(xticks)))
    
    # Full vertical grid aligned with ticks
    for x in xticks:
        if x == 0: continue
        ax.axvline(x, color="#e8e8e8", linewidth=0.6, zorder=1)

    # Horizontal separators between rows
    for i in range(n_y - 1):
        ax.axhline(i + 0.5, color="#cccccc", linewidth=0.6, zorder=2, alpha=0.3)

    ax.set_yticks(range(n_y))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_ylim(-0.6, n_y - 0.4)
    ax.invert_yaxis()

    ax.set_xlabel("Prediction Count", fontsize=10)
    ax.set_xlim(0, x_limit * 1.05)
    ax.set_xticks(xticks)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["left"].set_color("#aaaaaa")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("#aaaaaa")

    # Less technical title
    clean_task = task_name.replace("_", " ").title()
    fig.suptitle(
        f"Prediction Distribution: {synset_label}\nModel: {model_label} ({clean_task})",
        fontsize=12,
        y=0.98
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
