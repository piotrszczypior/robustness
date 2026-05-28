from __future__ import annotations
from dataclasses import dataclass, field
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
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        self._npy_path = base.with_suffix(".npy")
        self._parquet_path = base.with_suffix(".parquet")
        self._embeddings: list[np.ndarray] = []
        self._images: list[str] = []
        self._synsets: list[str] = []
        self._y_true: list[int] = []
        self._y_pred: list[int] = []

    def __enter__(self) -> EmbeddingWriter:
        self._embeddings.clear()
        self._images.clear()
        self._synsets.clear()
        self._y_true.clear()
        self._y_pred.clear()
        return self

    def __exit__(self, *_):
        if not self._embeddings:
            return
        np.save(self._npy_path, np.concatenate(self._embeddings, axis=0))
        pd.DataFrame(
            {
                "image": self._images,
                "synset": self._synsets,
                "y_true": self._y_true,
                "y_pred": self._y_pred,
            }
        ).to_parquet(self._parquet_path, index=False)

    def write_batch(
        self,
        filenames: tuple,
        synsets: tuple,
        targets: np.ndarray,
        predictions: np.ndarray,
        embeddings: np.ndarray,
    ) -> None:
        self._embeddings.append(embeddings)
        self._images.extend(filenames)
        self._synsets.extend(synsets)
        self._y_true.extend(targets.tolist())
        self._y_pred.extend(predictions.tolist())
