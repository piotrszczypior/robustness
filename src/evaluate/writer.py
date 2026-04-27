from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

__all__ = ["ResultAccumulator", "EmbeddingWriter"]


@dataclass
class ResultAccumulator:
    model_name: str
    image: list[str] = field(default_factory=list)
    synset: list[str] = field(default_factory=list)
    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)
    dataset_metadata: dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        filenames: tuple,
        synsets: tuple,
        targets: np.ndarray,
        predictions: np.ndarray,
        confidences: np.ndarray,
    ):
        self.image.extend(filenames)
        self.synset.extend(synsets)
        self.y_true.extend(targets)
        self.y_pred.extend(predictions)
        self.confidence.extend(confidences)

    def with_metadata(self, metadata: dict[str, Any]) -> ResultAccumulator:
        self.dataset_metadata = metadata
        return self

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "model": self.model_name,
                "image": self.image,
                "synset": self.synset,
                "y_true": self.y_true,
                "y_pred": self.y_pred,
                "confidence": self.confidence,
            }
        )
        df["is_correct"] = (df["y_true"] == df["y_pred"]).astype(int)

        for key, value in self.dataset_metadata.items():
            df[key] = value

        return df


class EmbeddingWriter:
    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def __enter__(self) -> EmbeddingWriter:
        self._file = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *_):
        if self._file:
            self._file.close()
            self._file = None

    def write_batch(
        self,
        filenames: tuple,
        synsets: tuple,
        targets: np.ndarray,
        predictions: np.ndarray,
        embeddings: np.ndarray,
    ) -> None:
        for image, synset, y_true, y_pred, vec in zip(
            filenames, synsets, targets, predictions, embeddings
        ):
            record = {
                "image": image,
                "synset": synset,
                "y_true": int(y_true),
                "y_pred": int(y_pred),
                "embedding": vec.tolist(),
            }
            self._file.write(json.dumps(record) + "\n")
