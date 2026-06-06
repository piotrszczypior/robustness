from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT             = Path(__file__).resolve().parents[1]
INPUT_JSON       = ROOT / "results" / "representations" / "attractors" / "set_attractors.json"
OUT_STEM         = ROOT / "results" / "representations" / "attractors" / "attractor_settings_distribution"
HIGHLIGHT_SYNSET = "adasda"
N_LABELED        = 8
COLOR_DEFAULT    = "#5B7FA6"
COLOR_HIGHLIGHT  = "#5B7FA6"
FIG_W_IN         = 14.0
FIG_H_IN         = 4.5
DPI              = 150
FONTSIZE_LABEL   = 8

# plt.rcParams.update({
#     "pdf.fonttype": 42,
#     "ps.fonttype":  42,
#     "font.size":    9,
# })


def main() -> None:
    with open(INPUT_JSON) as f:
        records: list[dict] = json.load(f)

    records.sort(key=lambda r: r["n_corruptions"], reverse=True)

    xs      = list(range(len(records)))
    heights = [r["n_corruptions"] for r in records]
    colors  = [COLOR_HIGHLIGHT if r["attractor_synset"] == HIGHLIGHT_SYNSET else COLOR_DEFAULT
               for r in records]

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    ax.bar(xs, heights, color=colors, width=0.8)

    ax.set_xlabel("Attractor classes ranked")
    ax.set_ylabel("Number of ImageNet-C corruptions")
    ax.set_xlim(-0.8, len(records) - 0.2)
    ax.set_yticks(range(0, 55, 1))
    ax.tick_params(bottom=False)
    ax.set_xticks([])
    ax.spines[["top", "right"]].set_visible(False)

    labeled_indices = sorted(set(range(min(N_LABELED, len(records)))))
    labeled_indices.append(15)
    highlight_idx   = next(
        (i for i, r in enumerate(records) if r["attractor_synset"] == HIGHLIGHT_SYNSET), None
    )
    if highlight_idx is not None and highlight_idx not in labeled_indices:
        labeled_indices.append(highlight_idx)
        labeled_indices.sort()

    px_per_bar  = FIG_W_IN * DPI / len(records)
    px_per_char = FONTSIZE_LABEL * (DPI / 72.0) * 0.58
    min_gap     = px_per_char * 20 / px_per_bar
    row_cursor  = [None, None]

    x_text_id = [0.5 ,0.5, 0.5, 0.5, 1, 1, 1.25, 1.25, 1.75]

    line_kw = dict(color="#cccccc", lw=0.7, zorder=1, clip_on=False)

    for rank, bar_i in enumerate(labeled_indices):
        y_bar  = heights[bar_i]
        if bar_i == 15:
            text_x = x_text_id[-1] * 10
        else:
            text_x = x_text_id[bar_i] * 10
        # if row_cursor[row] is not None:
        #     text_x = max(text_x, row_cursor[row] + min_gap)
        # row_cursor[row] = text_x

        rec   = records[bar_i]
        label = rec["attractor_label"].replace("_", " ")
        synset = rec["attractor_synset"].replace("_", " ")

        color = COLOR_HIGHLIGHT if rec["attractor_synset"] == HIGHLIGHT_SYNSET else "black"

        if bar_i == 0:
            ax.plot([bar_i, bar_i], [y_bar, y_bar + 2], **line_kw)
            ax.plot([bar_i, text_x], [y_bar + 2, y_bar + 2], **line_kw)
            ax.text(text_x, y_bar + 1.75, f"{label.capitalize()} ({synset})", ha="left", va="bottom",
                    fontsize=FONTSIZE_LABEL, color=color)
            continue
            
        ax.plot([bar_i, bar_i], [y_bar, y_bar + 1], **line_kw)
        ax.plot([bar_i, text_x], [y_bar + 1, y_bar + 1], **line_kw)
        ax.text(text_x, y_bar + 0.75, f"{label.capitalize()} ({synset})" , ha="left", va="bottom",
                fontsize=FONTSIZE_LABEL, color=color)


    fig.tight_layout()
    fig.savefig(str(OUT_STEM) + ".png", bbox_inches="tight", facecolor="white", dpi=DPI)
    plt.close(fig)
    print(f"Saved: {OUT_STEM}.png")


if __name__ == "__main__":
    main()
