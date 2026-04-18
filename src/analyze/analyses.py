from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from model import MODELS

from .base import BaseTask, ModelTest, VariantEntry, VariationSpace


@dataclass(frozen=True)
class FragileClassTask(BaseTask):
    type: Literal["fragile_class"] = field(default="fragile_class", init=False)

    baseline_csv: str = ""
    corrupted_csv: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.baseline_csv:
            raise ValueError(f"[{self.name}] baseline_csv must not be empty.")
        if not self.corrupted_csv:
            raise ValueError(f"[{self.name}] corrupted_csv must not be empty.")


@dataclass(frozen=True)
class AccuracyDropTask(BaseTask):
    type: Literal["accuracy_drop"] = field(default="accuracy_drop", init=False)

    baseline_csv: str = ""
    corrupted_csv: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.baseline_csv:
            raise ValueError(f"[{self.name}] baseline_csv must not be empty.")
        if not self.corrupted_csv:
            raise ValueError(f"[{self.name}] corrupted_csv must not be empty.")


@dataclass(frozen=True)
class CommonFragileClassTask(BaseTask):
    type: Literal["common_fragile_class"] = field(
        default="common_fragile_class", init=False
    )

    fragile_class_files: tuple[str, ...] = field(default_factory=tuple)
    output_filename: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.fragile_class_files:
            raise ValueError(f"[{self.name}] fragile_class_files must not be empty.")
        if not self.output_filename:
            raise ValueError(f"[{self.name}] output_filename must not be empty.")


@dataclass(frozen=True)
class FragileClassOverlapTask(BaseTask):
    type: str = "class_overlap_test"
    test_type: str
    models: tuple[ModelTest, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.test_type, "test type must not be empty"
        assert len(self.tests) >= 2, f"[{self.name}] at least 2 tests are required"


def name(var: VariantEntry) -> str:
    return f"{var.model}_{var.group}_{var.corruption}_{var.severity}"


def corrupted_csv(var: VariantEntry) -> str:
    return f"{var.model}_imagenet_c_{var.group}_{var.corruption}_{var.severity}.csv"


def baseline_csv(model: str) -> str:
    return f"{model}_imagenet.csv"


def classes_json(model: str, var: VariantEntry) -> str:
    name = f"{model}_{var.group}_{var.corruption}_{var.severity}"
    return f"{name}/{name}_classes.json"


def generate_fragile_class_tasks(
    space: VariationSpace | None = None,
) -> list[FragileClassTask]:
    """All tasks for calculating fragile classes"""

    return [
        FragileClassTask(
            name=name(variant),
            baseline_csv=baseline_csv(variant.model),
            corrupted_csv=corrupted_csv(variant),
        )
        for variant in (space or VariationSpace())
    ]


def generate_accuracy_drop_tasks(
    space: VariationSpace | None = None,
) -> list[AccuracyDropTask]:
    """All model x corruption x severity accuracy-drop tasks"""

    return [
        AccuracyDropTask(
            name=name(variant),
            baseline_csv=baseline_csv(variant.model),
            corrupted_csv=corrupted_csv(variant),
        )
        for variant in (space or VariationSpace())
    ]


def generate_fragile_class_overlap_tasks(
    space: VariationSpace | None = None,
    test_type: str = "chi2",
) -> list[FragileClassOverlapTask]:
    """Classes overlap-test"""

    return [
        FragileClassOverlapTask(
            name=f"ImageNet-C: {corruption.replace('_', ' ').title()} Severity {severity}",
            type=test_type,
            tests=tuple(
                ModelTest(
                    label=MODELS[model],
                    data=classes_json(
                        model, VariantEntry(model, group, corruption, severity)
                    ),
                )
                for model in models
            ),
        )
        for group, corruption, severity, models in (
            space or VariationSpace()
        ).per_corruption()
    ]


def generate_common_fragile_tasks(
    space: VariationSpace | None = None,
) -> list[CommonFragileClassTask]:
    """One cross-model intersection task per corruption x severity."""

    return [
        CommonFragileClassTask(
            name=f"Common Fragile Classes - {corruption} severity {severity}",
            fragile_class_files=tuple(
                classes_json(model, VariantEntry(model, group, corruption, severity))
                for model in models
            ),
            output_filename=f"common_fragile_{corruption}_{severity}.json",
        )
        for group, corruption, severity, models in (
            space or VariationSpace()
        ).per_corruption()
    ]
