from __future__ import annotations

from dataclasses import dataclass
import logging
import pandas as pd
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

__all__ = ["get_recipe", "Recipe"]


def get_recipe(name: str) -> Recipe:
    return _ChartRecipeRegistry.get_recipe(name)


@dataclass(frozen=True)
class Recipe:
    name: str
    groupby: Union[str, List[str]]
    column: str
    aggregate: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(self.groupby).agg(
            **{self.name: (self.column, self.aggregate)}
        )


class _ChartRecipeRegistry:
    _RECIPE_REGISTRY: Dict[str, Recipe] = {
        "accuracy": Recipe(
            name="accuracy", groupby="synset", column="is_correct", aggregate="mean"
        ),
        "mean-accuracy": Recipe(
            name="accuracy",
            groupby="severity",
            column="is_correct",
            aggregate="mean",
        ),
    }

    @classmethod
    def get_recipe(cls, name: str) -> Recipe:
        if name not in cls._RECIPE_REGISTRY:
            logger.error(f"[ERROR] Recipe '{name}' does not exist in registry!")
            raise KeyError(f"[ERROR]: Recipe '{name}' does not exist in registry!")
        return cls._RECIPE_REGISTRY[name]
