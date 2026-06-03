from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import pandas as pd

from space import CorruptionVariations

from .aggregate import aggregate_to_class
from .loader import load_aligned
from .metrics import PerImageMetrics, compute_per_image_metrics
from .naming import clean_name, condition_name

__all__ = [
    "ConditionResult",
    "CorruptionResult",
    "build_variations",
    "iter_condition_results",
    "iter_corruption_results",
    "run",
]

logger = logging.getLogger(__name__)


@dataclass
class ConditionResult:
    """Per-image metrics for a single (corruption, severity) vs clean."""

    model: str
    group: str
    corruption: str
    severity: int
    cond_stem: str
    metrics: PerImageMetrics  # .table (scalars) + .delta (raw)
    class_table: pd.DataFrame  # (n_classes, ~5) per-synset aggregation (1.3)


@dataclass
class CorruptionResult:
    """All severities of one corruption, grouped together."""

    model: str
    group: str
    corruption: str
    conditions: list[ConditionResult] = field(default_factory=list)


def build_variations(
    model: str,
    groups: list[str] | None,
    corruptions: list[str] | None,
    severities: list[int] | None,
) -> CorruptionVariations:
    """Build the corruption space to iterate over for a single model."""
    return CorruptionVariations(
        models=[model],
        groups=groups,
        corruptions=corruptions,
        severities=severities,
    )


def iter_condition_results(
    model: str,
    variations: CorruptionVariations,
    embeddings_dir: Path | str | None = None,
) -> Generator[ConditionResult, None, None]:
    """Stream 1.1–1.3 once per (corruption, severity) condition.

    Yields one ConditionResult at a time. Because nothing is accumulated here,
    each condition's raw ``delta`` matrix (~400 MB) is freed before the next is
    loaded — this is what makes the 75-condition roll-up fit in memory.
    Missing embedding files are skipped with a warning.
    """
    clean = clean_name(model)

    for group, corruption, severities in variations.per_unique_corruption():
        for severity in severities:
            cond_stem = condition_name(model, group, corruption, severity)
            try:
                aligned = load_aligned(clean, cond_stem, embeddings_dir)
            except FileNotFoundError as exc:
                logger.warning("Skipping %s sev%d: %s", corruption, severity, exc)
                continue

            metrics = compute_per_image_metrics(aligned)
            class_table = aggregate_to_class(metrics)
            yield ConditionResult(
                model=model,
                group=group,
                corruption=corruption,
                severity=severity,
                cond_stem=cond_stem,
                metrics=metrics,
                class_table=class_table,
            )


def iter_corruption_results(
    model: str,
    variations: CorruptionVariations,
    embeddings_dir: Path | str | None = None,
) -> Generator[CorruptionResult, None, None]:
    """Group :func:`iter_condition_results` by corruption.

    Note: this retains every condition (and its delta) of a corruption at once.
    Fine for small selections; for the full roll-up prefer the streaming
    :func:`iter_condition_results` / :func:`representations.dataset.build_tidy`.
    """
    current: CorruptionResult | None = None

    for condition in iter_condition_results(model, variations, embeddings_dir):
        if current is None or current.corruption != condition.corruption:
            if current is not None:
                yield current
            current = CorruptionResult(
                model=condition.model,
                group=condition.group,
                corruption=condition.corruption,
            )
        current.conditions.append(condition)

    if current is not None:
        yield current


def run(
    model: str,
    groups: list[str] | None = None,
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
    embeddings_dir: Path | str | None = None,
) -> list[CorruptionResult]:
    """Entry point: iterate the corruption space and compute per-image metrics.

    Returns the collected results so callers (and later pipeline steps) can keep
    working with them. This is the template that steps 1.3+ build on top of.
    """
    variations = build_variations(model, groups, corruptions, severities)
    logger.info(
        "Running representation analysis for model=%s over %d variants",
        model,
        len(variations),
    )

    results: list[CorruptionResult] = []
    for corruption_result in iter_corruption_results(
        model=model,
        variations=variations,
        embeddings_dir=embeddings_dir,
    ):
        _summarize_corruption(corruption_result)

        # ------------------------------------------------------------------
        # TODO 1.4+: plug further analysis here. Each ConditionResult carries
        #   - .metrics.table : per-image scalar metrics (+ synset)
        #   - .metrics.delta : raw (N, D) displacement matrix (kept on purpose)
        #   - .class_table   : per-synset median/IQR + coherence R (1.3)
        # Likely extensions:
        #   * shared displacement subspace per corruption (PCA on delta)
        #   * cross-severity trends, cross-corruption comparison
        #   * ranking classes by low coherence / high relative_shift
        # ------------------------------------------------------------------

        results.append(corruption_result)

    return results


def _summarize_corruption(result: CorruptionResult) -> None:
    print(f"\n[{result.model}] {result.group}/{result.corruption}")
    for condition in result.conditions:
        classes = condition.class_table
        print(
            f"  sev{condition.severity}: "
            f"classes={len(classes)} "
            f"rel_shift_med={classes['relative_shift_median'].median():.4f} "
            f"ang_dist_med={classes['angular_distance_median'].median():.4f} "
            f"coherence={classes['coherence'].mean():.4f}"
        )
