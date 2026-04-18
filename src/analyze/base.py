from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Generator

from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES
from model import MODELS


@dataclass(frozen=True)
class BaseTask:
    name: str
    type: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Task name must not be empty.")


@dataclass(frozen=True)
class ModelTest:
    model: str
    data: str

    def __post_init__(self) -> None:
        assert self.model, "ModelTest.model must not be empty"
        assert self.data, "ModelTest.data must not be empty"


@dataclass(frozen=True)
class VariantEntry:
    model: str
    group: str
    corruption: str
    severity: int


@dataclass(frozen=True)
class VariationSpace:
    models: list[str] | None = None
    groups: list[str] | None = None
    severities: list[int] | None = None

    _variants: tuple[VariantEntry, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _models = self.models or list(MODELS)
        _severities = self.severities or IMAGENET_C_SEVERITIES
        _groups = {
            group: corruptions
            for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items()
            if group in (self.groups or IMAGENET_C_CORRUPTION_GROUPS)
        }
        variants = [
            VariantEntry(
                model=model, group=group, corruption=corruption, severity=severity
            )
            for model, (group, corruptions), severity in product(
                _models, _groups.items(), _severities
            )
            for corruption in corruptions
        ]
        object.__setattr__(self, "_variants", tuple(variants))

    def __iter__(self) -> Generator[VariantEntry, None, None]:
        yield from self._variants

    def __len__(self) -> int:
        return len(self._variants)

    def per_corruption(self) -> Generator[tuple[str, str, int, list[str]], None, None]:
        seen: dict[tuple[str, str, int], list[str]] = {}
        for v in self._variants:
            key = (v.group, v.corruption, v.severity)
            seen.setdefault(key, []).append(v.model)

        for (group, corruption, severity), models in seen.items():
            yield group, corruption, severity, models
