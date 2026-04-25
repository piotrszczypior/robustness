from __future__ import annotations

import json
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd


from model import MODELS


MODEL_COLORS = {
    "convnext_base": "#534AB7",
    "efficientnet_b4": "#1D9E75",
    "resnet152": "#D85A30",
    "resnet50": "#BA7517",
    "vit_b_16": "#D4537E",
}

SYNSET_COLORS = [
    "#378ADD",
    "#1D9E75",
    "#D85A30",
    "#BA7517",
    "#D4537E",
]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def load_human_readable_labels():
    path = Path("imagenet_class_index.json")

    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient="index", columns=["synset", "label"])
    df.index = df.index.astype(int)
    df = df.sort_index()

    return df


def build_sankey_per_synset(
    data: list[dict], synset: str, min_count: int = 1, title: str = None
) -> go.Figure:
    data = [d for d in data if d["count"] > min_count]
    index_to_synset = load_human_readable_labels()

    models = sorted(set(d["model"] for d in data))
    ypreds = sorted(set(d["y_pred"] for d in data))

    model_idx = {m: i for i, m in enumerate(models)}
    ypred_idx = {y: len(models) + i for i, y in enumerate(ypreds)}

    synset_row = index_to_synset[index_to_synset["synset"] == synset].iloc[0]
    synset_label = f"{synset} ({synset_row['label']})"

    node_labels = [MODELS[m] for m in (models)] + [
        f"{index_to_synset.loc[y, 'synset']} ({index_to_synset.loc[y, 'label']})"
        if y in index_to_synset.index
        else str(y)
        for y in ypreds
    ]
    node_colors = [MODEL_COLORS.get(m, "#888888") for m in models] + ["#7F77DD"] * len(
        ypreds
    )

    sources, targets, values, link_colors = [], [], [], []
    for d in data:
        sources.append(model_idx[d["model"]])
        targets.append(ypred_idx[d["y_pred"]])
        values.append(d["count"])
        link_colors.append(hex_to_rgba(MODEL_COLORS.get(d["model"], "#888888"), 0.35))

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=30,
                line=dict(color="rgba(0,0,0,0.15)", width=0.5),
                label=node_labels,
                color=node_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
            ),
        )
    )

    fig.update_layout(
        title_text=f"{title}. Predictions for synset: {synset_label}",
        font_size=12,
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def generate_sankey_plots(
    input_file: Path, output_dir: Path, min_count: int, title: str = None
):
    with open(input_file) as f:
        payload = json.load(f)

    data = payload["weighted_preds"]
    metadata = payload["metadata"]
    synsets = sorted(set(d["synset"] for d in data))

    output_dir.mkdir(parents=True, exist_ok=True)

    for synset in synsets:
        synset_data = [d for d in data if d["synset"] == synset]
        fig = build_sankey_per_synset(
            synset_data, synset, min_count=min_count, title=title
        )

        filename = f"sankey_{synset}_{metadata['task_name']}_min_count_{min_count}"

        png_path = output_dir / f"{filename}.png"
        fig.write_image(str(png_path), width=1200, height=600, scale=2)
