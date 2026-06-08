from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class FragileDefinition:
    name: str
    label: str
    combine: Callable[[pd.DataFrame], "pd.Series[bool]"]


def _and_ab(df: pd.DataFrame) -> "pd.Series[bool]":
    return (df["is_fragile_a"] == 1) & (df["is_fragile_b"] == 1)


def _b_and_a_or_c(df: pd.DataFrame) -> "pd.Series[bool]":
    return (df["is_fragile_b"] == 1) & (
        (df["is_fragile_a"] == 1) | (df["is_fragile_c"] == 1)
    )


def _and_abc(df: pd.DataFrame) -> "pd.Series[bool]":
    return (
        (df["is_fragile_a"] == 1)
        & (df["is_fragile_b"] == 1)
        & (df["is_fragile_c"] == 1)
    )


DEFINITIONS: dict[str, FragileDefinition] = {
    "ab": FragileDefinition("ab", "A ∩ B", _and_ab),
    "b_aoc": FragileDefinition("b_aoc", "B ∩ (A U+222a C)", _b_and_a_or_c),
    "abc": FragileDefinition("abc", "A ∩ B ∩ C", _and_abc),
}
