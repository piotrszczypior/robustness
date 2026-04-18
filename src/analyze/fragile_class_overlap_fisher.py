from __future__ import annotations

from scipy.stats import fisher_exact
from itertools import combinations
import numpy as np
import pandas as pd

from .settings import BaseAnalysisConfig


def run(config: BaseAnalysisConfig, output_dir: str):
    content = config.content
    tests = content["tests"]
    loaded_data = {item["label"]: _get_data(item["data"]) for item in tests}
    labels = [item["label"] for item in tests]
    test_pairs = list(combinations(labels, 2))

    all_results = []

    for l1, l2 in test_pairs:
        res_dict = calculate_fisher_for_pair(
            loaded_data[l1]["is_fragile"].values, loaded_data[l2]["is_fragile"].values
        )
        res_dict["Comparison"] = f"{l1} vs {l2}"
        all_results.append(res_dict)

    df_results = pd.DataFrame(all_results)

    cols = ["Comparison", "Shared_Fragile", "Odds_Ratio", "p-value", "Significant"]
    df_results = df_results[cols]

    print(df_results)

    results_list = []
    for l1, l2 in test_pairs:
        stats = calculate_fisher_for_pair(
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


def calculate_fisher_for_pair(list_a, list_b):
    v1, v2 = np.array(list_a), np.array(list_b)

    a = np.sum((v1 == 1) & (v2 == 1))
    b = np.sum((v1 == 1) & (v2 == 0))
    c = np.sum((v1 == 0) & (v2 == 1))
    d = np.sum((v1 == 0) & (v2 == 0))

    table = [[a, b], [c, d]]
    odds_ratio, p_value = fisher_exact(table)

    return {
        "Odds_Ratio": round(odds_ratio, 4),
        "p-value": float(f"{p_value:.4e}"),
        "Shared_Fragile": int(a),
        "Significant": p_value < 0.05,
    }
