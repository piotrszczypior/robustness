from pathlib import Path

import matplotlib.pyplot as plt

_COLOR_CORRECT = "#2980B9"
_COLOR_MISTAKE = "#000000"


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
    transpose: bool = False,
) -> None:
    if not entries:
        print(f"  No data for {synset}, skipping.")
        return

    if transpose:
        _render_transposed(entries, synset, synset_label, model_label, output_path)
    else:
        _render_horizontal(entries, synset, synset_label, model_label, output_path)


def _render_horizontal(
    entries: list[dict],
    synset: str,
    synset_label: str,
    model_label: str,
    output_path: Path,
) -> None:
    y_labels = [f"{e['label']}\n({e['synset']})" for e in entries]
    counts = [e['count'] for e in entries]
    is_correct_list = [e['is_correct'] for e in entries]
    n_y = len(entries)
    x_max = max(counts)
    x_limit = max(50, x_max)

    fig_h = max(3, n_y * 0.9) + 1.0
    fig, ax = plt.subplots(figsize=(max(8, x_max * 0.1 + 4), fig_h), dpi=150)
    fig.patch.set_facecolor("white")

    for i, (count, is_correct) in enumerate(zip(counts, is_correct_list)):
        color = _COLOR_CORRECT if is_correct else _COLOR_MISTAKE
        ax.plot([0, count], [i, i], color=color, linewidth=1.5, solid_capstyle="round", zorder=2)
        if is_correct:
            ax.plot(count, i, "o", color=color, markersize=6, zorder=3)
        else:
            ax.plot(count, i, "o", color=color, markersize=6,
                    markerfacecolor="white", markeredgewidth=1.5, zorder=3)

    xticks = list(range(0, int(x_limit) + 1, 5))
    if 50 not in xticks:
        xticks.append(50)
    xticks = sorted(list(set(xticks)))

    ax.set_yticks(range(n_y))
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_ylim(n_y - 0.5, -0.5)

    ax.set_xlabel("Prediction Count", fontsize=14)
    ax.set_xlim(0, x_limit * 1.05)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=12)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(color="#eeeeee", linewidth=1, zorder=0)

    ax.set_title(f"{synset_label}  ·  {model_label}", fontsize=14, pad=6)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # fig.savefig(output_path, dpi=150, bbox_inches="tight")
    from utils import save_as_pdf
    save_as_pdf(fig, output_path)
    plt.close(fig)


def _render_transposed(
    entries: list[dict],
    synset: str,
    synset_label: str,
    model_label: str,
    output_path: Path,
) -> None:
    x_labels = [f"{e['label']}\n({e['synset']})" for e in entries]
    counts = [e['count'] for e in entries]
    is_correct_list = [e['is_correct'] for e in entries]
    n = len(entries)
    y_max = max(counts)

    tick_step = max(1, round(y_max / 8))
    yticks = list(range(0, y_max + tick_step + 1, tick_step))

    fig, ax = plt.subplots(figsize=(max(8, n * 1.5 + 2), 5), dpi=150)
    fig.patch.set_facecolor("white")

    for i, (count, is_correct) in enumerate(zip(counts, is_correct_list)):
        color = _COLOR_CORRECT if is_correct else _COLOR_MISTAKE
        ax.vlines(i, 0, count, color=color, linewidth=1.5, zorder=2)
        if is_correct:
            ax.plot(i, count, "o", color=color, markersize=6, zorder=3)
        else:
            ax.plot(i, count, "o", color=color, markersize=6,
                    markerfacecolor="white", markeredgewidth=1.5, zorder=3)

    ax.set_xticks(range(n))
    ax.set_xticklabels(x_labels, fontsize=12, rotation=30, ha="right")
    ax.set_xlim(-0.6, n - 0.4)

    ax.set_ylabel("Prediction Count", fontsize=14)
    ax.set_ylim(0, y_max * 1.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks], fontsize=12)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=1, zorder=0)
    ax.tick_params(bottom=True, length=4)

    ax.set_title(f"{synset_label}  ·  {model_label}", fontsize=14, pad=6)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # fig.savefig(output_path, dpi=150, bbox_inches="tight")
    from utils import save_as_pdf
    save_as_pdf(fig, output_path)
    plt.close(fig)
