from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from analyze.analyses import FragileClassOverlapTask

logger = logging.getLogger(__name__)


def run(task: FragileClassOverlapTask, output_dir: str) -> None:
    _FragileClassOverlapAnalysis(task, output_dir).run()


def _contingency_table(
    list_a: np.ndarray, list_b: np.ndarray
) -> tuple[int, int, int, int]:
    v1, v2 = np.array(list_a), np.array(list_b)
    a = int(np.sum((v1 == 1) & (v2 == 1)))
    b = int(np.sum((v1 == 1) & (v2 == 0)))
    c = int(np.sum((v1 == 0) & (v2 == 1)))
    d = int(np.sum((v1 == 0) & (v2 == 0)))

    return a, b, c, d


def _chi2_for_pair(list_a: np.ndarray, list_b: np.ndarray) -> dict:
    a, b, c, d = _contingency_table(list_a, list_b)
    chi2, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=True)

    return {
        "Shared_Fragile": a,
        "Chi2_Stat": round(float(chi2), 4),
        "p-value": float(f"{p:.4e}"),
        "Significant": p < 0.05,
    }


def _fisher_for_pair(list_a: np.ndarray, list_b: np.ndarray) -> dict:
    a, b, c, d = _contingency_table(list_a, list_b)
    odds_ratio, p = fisher_exact([[a, b], [c, d]])

    return {
        "Shared_Fragile": a,
        "Odds_Ratio": round(float(odds_ratio), 4),
        "p-value": float(f"{p:.4e}"),
        "Significant": p < 0.05,
    }


class _FragileClassOverlapAnalysis:
    _STAT_FN: dict[str, Callable[[np.ndarray, np.ndarray], dict]] = {
        "chi2": _chi2_for_pair,
        "fisher": _fisher_for_pair,
    }

    def __init__(self, task: FragileClassOverlapTask, output_dir: str) -> None:
        self.task = task
        self.output_dir = Path(output_dir)
        self.stat_fn = self._STAT_FN[task.test_type]

    def run(self) -> None:
        logger.info(f"Running analysis: '{self.task.name}' [{self.task.type}]")

        loaded_data = {
            test.label: self._load_fragile_data(test.data) for test in self.task.tests
        }

        results = [
            {
                "Comparison": f"{l1} vs {l2}",
                **self.stat_fn(
                    loaded_data[l1]["is_fragile"].values,
                    loaded_data[l2]["is_fragile"].values,
                ),
            }
            for l1, l2 in combinations(loaded_data.keys(), 2)
        ]

        df = pd.DataFrame(results)
        df = df[["Comparison"] + [c for c in df.columns if c != "Comparison"]]

        logger.info(f"\n{df.to_markdown(index=False)}")

    def _load_fragile_data(self, filename: str) -> pd.DataFrame:
        logger.info(f"Loading fragile class data from: {filename}")
        path = Path("analysis/results") / filename
        with open(path, "r") as f:
            data = json.load(f)
        return pd.json_normalize(data, record_path=["classes"], meta=["name"])
