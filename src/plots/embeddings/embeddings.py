from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import numpy as np

from plots.base import BasePlotPipeline
from plots.specs import ChartConfig


class EmbeddingProjector(BasePlotPipeline):
    def __init__(self, config: ChartConfig, data_dir: Path | str):
        super().__init__(config, data_dir)
        with open("imagenet_class_index.json") as f:
            self.class_index = {v[0]: v[1] for k, v in json.load(f).items()}

    def transform_data(self):
        data_frames = []
        for source in self.config.content.files:
            filepath = self.data_dir / source["file"]
            df = pd.read_json(filepath, lines=True)
            df["source"] = source["label"]
            data_frames.append(df)

        data = pd.concat(data_frames)
        selected_classes = self.config.content.classes
        data = data[data["synset"].isin(selected_classes)]
        return data

    def render(self, data: pd.DataFrame):
        embeddings = np.vstack(data["embedding"].values)

        if self.config.content.projection == "pca":
            reducer = PCA(n_components=2)
        elif self.config.content.projection == "tsne":
            reducer = TSNE(n_components=2, perplexity=min(30, len(data) - 1))
        elif self.config.content.projection == "umap":
            reducer = umap.UMAP(n_components=2)
        else:
            raise ValueError(
                f"Unknown reduction method: {self.config.content.projection}"
            )

        projected_embeddings = reducer.fit_transform(embeddings)
        data["x"] = projected_embeddings[:, 0]
        data["y"] = projected_embeddings[:, 1]

        for name, group in data.groupby("synset"):
            label = self.class_index.get(name, name)
            self.ax.scatter(group["x"], group["y"], label=label, alpha=0.7)

        self.ax.legend()
