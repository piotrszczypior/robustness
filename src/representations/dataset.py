from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from space import CorruptionVariations

from .aggregate import CLASS_COLUMNS
from .runner import build_variations, iter_condition_results

__all__ = [
    "TIDY_ID_COLUMNS",
    "build_tidy",
    "save_tidy",
]

logger = logging.getLogger(__name__)

# Identifying columns of the tidy table; everything else is (metric, value).
TIDY_ID_COLUMNS = ["model", "group", "corruption", "severity", "synset", "n"]

# Metric columns that get melted into long form (everything except "n").
_METRIC_COLUMNS = [c for c in CLASS_COLUMNS if c != "n"]


def build_tidy(
    model: str,
    groups: list[str] | None = None,
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
    embeddings_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Roll 1.1–1.3 up over every condition into one long (tidy) table.

    One row per (synset, corruption, severity, metric). This is the single
    dataset that every downstream figure reads from.

    Memory stays bounded: conditions are streamed via
    :func:`representations.runner.iter_condition_results`, so only one raw delta
    matrix is alive at a time; we keep just the small per-class tables.
    """
    variations: CorruptionVariations = build_variations(
        model, groups, corruptions, severities
    )
    logger.info(
        "Building tidy roll-up for model=%s over %d variants", model, len(variations)
    )

    frames: list[pd.DataFrame] = []
    n_conditions = 0
    for condition in iter_condition_results(model, variations, embeddings_dir):
        frames.append(_tag_class_table(condition))
        n_conditions += 1

    if not frames:
        raise FileNotFoundError(
            f"No conditions produced any data for model '{model}'. "
            "Check that the clean baseline and condition embeddings exist."
        )

    wide = pd.concat(frames, ignore_index=True)
    tidy = wide.melt(
        id_vars=TIDY_ID_COLUMNS,
        value_vars=_METRIC_COLUMNS,
        var_name="metric",
        value_name="value",
    )

    logger.info(
        "Tidy roll-up: %d conditions, %d synsets, %d metrics -> %d rows",
        n_conditions,
        tidy["synset"].nunique(),
        tidy["metric"].nunique(),
        len(tidy),
    )
    return tidy


def save_tidy(tidy: pd.DataFrame, out_path: Path | str) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        tidy.to_csv(path, index=False)
    else:
        tidy.to_parquet(path, index=False)
    logger.info("Saved tidy dataset (%d rows) to %s", len(tidy), path)
    return path


def _tag_class_table(condition) -> pd.DataFrame:
    table = condition.class_table.reset_index()  # synset becomes a column
    table.insert(0, "severity", condition.severity)
    table.insert(0, "corruption", condition.corruption)
    table.insert(0, "group", condition.group)
    table.insert(0, "model", condition.model)
    return table
