from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .metrics import PerImageMetrics

__all__ = [
    "SCALAR_METRICS",
    "CLASS_COLUMNS",
    "aggregate_to_class",
    "directional_coherence",
]

logger = logging.getLogger(__name__)

# Scalars from the 1.2 table that get a robust median + IQR per class.
SCALAR_METRICS = [
    "relative_shift",
    "angular_distance",
    "tangential_fraction",
]

CLASS_COLUMNS = (
    [f"{m}_median" for m in SCALAR_METRICS]
    + [f"{m}_iqr" for m in SCALAR_METRICS]
    + ["coherence", "n"]
)


def aggregate_to_class(metrics: PerImageMetrics) -> pd.DataFrame:
    """Aggregate per-image metrics to one row per class (synset).

    Two complementary parts:

    * scalar — group the 1.2 table by synset and take the median and IQR
      (robust to the ~50 per-class outliers) of each scalar metric;
    * vectorial — directional coherence R of the displacement, see
      :func:`directional_coherence`.

    Returns a DataFrame indexed by ``synset`` with median/IQR columns, the
    coherence ``R``, and the per-class sample count ``n``.
    """
    table = metrics.table
    synsets = table["synset"].to_numpy()

    grouped = table.groupby("synset")[SCALAR_METRICS]
    median = grouped.median()
    q1 = grouped.quantile(0.25)
    q3 = grouped.quantile(0.75)
    iqr = q3 - q1

    scalar = median.add_suffix("_median").join(iqr.add_suffix("_iqr"))

    coherence = directional_coherence(metrics.delta, synsets)
    counts = grouped.size().rename("n")

    result = scalar.join(coherence).join(counts)
    result = result[CLASS_COLUMNS]

    logger.info(
        "Aggregated %s vs %s to %d classes (median rel_shift=%.4f, mean R=%.4f)",
        metrics.clean_name,
        metrics.cond_name,
        len(result),
        float(result["relative_shift_median"].median()),
        float(result["coherence"].mean()),
    )
    return result


def directional_coherence(delta: np.ndarray, synsets: np.ndarray) -> pd.Series:
    """Per-class directional coherence R = ‖mean of unit displacements‖ ∈ [0, 1].

    Each row of ``delta`` is normalized to unit length (purely directional, so
    magnitude does not leak in), then averaged within a class. R ≈ 1 means the
    corruption pushes the whole class in one direction; R ≈ 0 means per-image
    chaos. Zero-length displacements contribute a zero vector.
    """
    norms = np.linalg.norm(delta, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    unit = (delta / safe[:, None]).astype(np.float64, copy=False)
    unit[norms == 0] = 0.0

    codes, uniques = pd.factorize(synsets)
    n_classes = len(uniques)

    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    sorted_unit = unit[order]

    starts = np.searchsorted(sorted_codes, np.arange(n_classes))
    sums = np.add.reduceat(sorted_unit, starts, axis=0)
    counts = np.diff(np.append(starts, len(sorted_codes)))

    mean_unit = sums / counts[:, None]
    resultant = np.linalg.norm(mean_unit, axis=1)

    return pd.Series(resultant, index=pd.Index(uniques, name="synset"), name="coherence")
