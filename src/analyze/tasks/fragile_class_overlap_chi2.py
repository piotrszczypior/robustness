from __future__ import annotations

from dataclasses import dataclass
from scipy.stats import chi2_contingency
from itertools import combinations
import numpy as np
import pandas as pd


from .settings import BaseAnalysisConfig


def run(config: BaseAnalysisConfig, output_dir: str):
    content = config.content
    tests = content["tests"]
    loaded_data = {item["label"]: _get_data(item["data"]) for item in tests}
    labels = [item["label"] for item in tests]
    tests = list(combinations(labels, 2))

    for l1, l2 in tests:
        print(l1, l2)
        result = pd.DataFrame(
            calculate_chi2_for_pair(
                loaded_data[l1]["is_fragile"].values,
                loaded_data[l2]["is_fragile"].values,
            ),
            index=[0],
        )
        print(result)
        print()

    results_list = []
    for l1, l2 in tests:
        stats = calculate_chi2_for_pair(
            loaded_data[l1]["is_fragile"].values,
            loaded_data[l2]["is_fragile"].values,
        )
        temp_df = pd.DataFrame(stats, index=[0])
        temp_df.insert(0, "Comparison", f"{l1} - {l2}")
        results_list.append(temp_df)

    final_df = pd.concat(results_list, ignore_index=True)
    print(final_df.to_markdown(index=False))


def _get_data(filename: str) -> pd.DataFrame:
    import json
    from pathlib import Path

    path = Path("analysis/results") / filename

    with open(path, "r") as f:
        data = json.load(f)

    return pd.json_normalize(data, record_path=["classes"], meta=["name"])


@dataclass
class Test:
    label: str
    data: str


@dataclass
class FragileClassTestConfig:
    name: str
    tests: list[Test]


def calculate_chi2_for_pair(list_a, list_b):
    v1, v2 = np.array(list_a), np.array(list_b)

    a = np.sum((v1 == 1) & (v2 == 1))
    b = np.sum((v1 == 1) & (v2 == 0))
    c = np.sum((v1 == 0) & (v2 == 1))
    d = np.sum((v1 == 0) & (v2 == 0))

    table = [[a, b], [c, d]]

    chi2, p, _, _ = chi2_contingency(table, correction=True)

    return {
        "Chi2_Stat": round(chi2, 4),
        "p-value": p,
        "Shared_Fragile_Count": a,
        "Significant": p < 0.05,
    }
