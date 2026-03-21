from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import pandas as pd
from typing import Any, Dict, Iterator, List, Union
import yaml

logger = logging.getLogger(__name__)


__all__ = ["register_recipes", "get_recipe", "Recipe"]


def register_recipes(recipes_file_path: Union[str, Path]):
    recipe_path = Path(recipes_file_path)

    if not recipe_path.exists():
        logger.error(f"Recipe file not found at: {recipe_path}")
        raise FileNotFoundError(f"Required recipe file {recipe_path} is missing!")

    recipes = list(_ChartRecipeFactory.from_yaml(recipe_path))
    for i, recipe in enumerate(recipes):
        logger.info(f"- Registering recipe {i}: {recipe.name}")
        _ChartRecipeRegistry.register_recipe(recipe)

    logger.info(f"Total of {len(recipes)} has been registered")


def get_recipe(name: str) -> Recipe:
    return _ChartRecipeRegistry.get_recipe(name)


@dataclass(frozen=True)
class Recipe:
    name: str
    type: str
    groupby: Union[str, List[str]]
    column: str
    aggregate: str

    def apply(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby(self.groupby)[self.column].agg(self.aggregate)


class _ChartRecipeRegistry:
    _RECIPE_REGISTRY: Dict[str, Recipe] = {}

    @classmethod
    def register_recipe(cls, recipe: Recipe):
        cls._RECIPE_REGISTRY[recipe.name] = recipe

    @classmethod
    def get_recipe(cls, name: str) -> Recipe:
        if name not in cls._RECIPE_REGISTRY:
            logger.error(f"Recipe '{name}' does not exists in registry!")
            raise KeyError(f"ERROR: Recipe '{name}' does not exists in registry!")

        return cls._RECIPE_REGISTRY[name]


class _ChartRecipeFactory:
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> Iterator[Recipe]:
        with open(yaml_path, "r") as f:
            contents = yaml.safe_load(f)

        return cls._from_dict(contents)

    @classmethod
    def _from_dict(cls, content: Dict[str, Any]) -> Iterator[Recipe]:
        recipes = content.get("recipes", [])

        for recipe in recipes:
            yield Recipe(**recipe)
