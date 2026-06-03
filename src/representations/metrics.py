from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loader import Features

__all__ = [
    "PerImageMetrics",
    "METRIC_COLUMNS",
    "compute_per_image_metrics",
]

logger = logging.getLogger(__name__)

METRIC_COLUMNS = [
    "clean_norm",
    "delta_norm",
    "relative_shift",
    "angular_distance",
    "tangential_fraction",
]


@dataclass
class PerImageMetrics:
    """Per-image displacement metrics for a single clean/condition pair.

    `delta` (F_cond - F_clean) is deliberately kept around: step 1.3 reuses the
    raw displacement matrix and recomputing it would mean re-reading the .npy.
    """

    table: pd.DataFrame  # (N, len(METRIC_COLUMNS) + 1) incl. "synset"
    delta: np.ndarray  # (N, D) float32, F_cond - F_clean
    clean_name: str
    cond_name: str

    @property
    def n(self) -> int:
        return int(self.delta.shape[0])


def compute_per_image_metrics(aligned: Features) -> PerImageMetrics:
    """Compute per-row displacement metrics from aligned feature matrices.

    Fully vectorized. Returns the metric table plus the raw delta matrix.
    """
    f_clean = aligned.clean_features
    f_corrupt = aligned.corrupt_features

    delta = (f_corrupt - f_clean).astype(np.float32, copy=False)

    # Metrics in float64 for numerical stability (norms of 2048-d float32 vecs).
    f_clean64 = f_clean.astype(np.float64, copy=False)
    f_cond64 = f_corrupt.astype(np.float64, copy=False)
    delta64 = delta.astype(np.float64, copy=False)

    clean_norm = np.linalg.norm(f_clean64, axis=1)
    cond_norm = np.linalg.norm(f_cond64, axis=1)
    delta_norm = np.linalg.norm(delta64, axis=1)

    safe_clean = np.where(clean_norm > 0, clean_norm, 1.0)
    relative_shift = np.where(clean_norm > 0, delta_norm / safe_clean, 0.0)

    denom = clean_norm * cond_norm
    cos_sim = np.where(denom > 0, np.einsum("ij,ij->i", f_clean64, f_cond64) / np.where(denom > 0, denom, 1.0), 1.0)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    cosine_distance = 1.0 - cos_sim

    # Decompose delta into radial (along clean) and tangential (perpendicular).
    # radial_sq = (delta . clean)^2 / ||clean||^2
    proj = np.einsum("ij,ij->i", delta64, f_clean64)
    radial_sq = np.where(clean_norm > 0, proj**2 / np.where(clean_norm > 0, clean_norm**2, 1.0), 0.0)
    delta_sq = delta_norm**2
    tangential_sq = np.clip(delta_sq - radial_sq, 0.0, None)
    tangential_fraction = np.where(delta_sq > 0, tangential_sq / np.where(delta_sq > 0, delta_sq, 1.0), 0.0)

    table = pd.DataFrame(
        {
            "clean_norm": clean_norm,
            "delta_norm": delta_norm,
            "relative_shift": relative_shift,
            "angular_distance": cosine_distance,
            "tangential_fraction": tangential_fraction,
            "synset": aligned.synsets,
        }
    )

    logger.info(
        "Per-image metrics for %s vs %s: N=%d, mean relative_shift=%.4f",
        aligned.clean_name,
        aligned.cond_name,
        len(table),
        float(table["relative_shift"].mean()),
    )

    return PerImageMetrics(
        table=table,
        delta=delta,
        clean_name=aligned.clean_name,
        cond_name=aligned.cond_name,
    )
