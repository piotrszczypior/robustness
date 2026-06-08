import pandas as pd
import numpy as np



def _pareto_indices_min_max(
    df: pd.DataFrame, maximize_col: str, minimize_col: str
) -> pd.Index:
    """Return index of non-dominated rows (maximize `maximize_col`, minimize `minimize_col`)."""
    hi = df[maximize_col].values
    lo = df[minimize_col].values
    n = len(hi)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if hi[j] >= hi[i] and lo[j] <= lo[i] and (hi[j] > hi[i] or lo[j] < lo[i]):
                dominated[i] = True
                break
    return df.index[~dominated]


def pareto_indices_max_max(
    df: pd.DataFrame, first_col: str, second_col: str
) -> pd.Index:
    """Return index of non-dominated rows (maximize both `first_col` and `second_col`)."""
    a = df[first_col].values
    b = df[second_col].values
    n = len(a)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if a[j] >= a[i] and b[j] >= b[i] and (a[j] > a[i] or b[j] > b[i]):
                dominated[i] = True
                break

    return df.index[~dominated]