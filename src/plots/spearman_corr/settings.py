from __future__ import annotations

from typing import Iterator
from plots.specs import ChartConfig
from space import CorruptionVariations
from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES
from model import MODELS


def construct_plot_config(space: CorruptionVariations, metric_type: str = "drop"):
    from munch import DefaultMunch

    for group, corruption, severity, models in space.per_corruption():
        models_content = []
        for model in models:
            model_slug = model.lower().replace("-", "")
            models_content.append(
                {
                    "name": model,
                    "clean": f"{model_slug}_imagenet.csv",
                    "corrupted": f"{model_slug}_imagenet_c_{group}_{corruption}_{severity}.csv",
                }
            )

        content = {
            "models": models_content,
            "metric_type": metric_type,
        }

        yield ChartConfig(
            name=f"imagenet_c_{corruption}_{severity}_spearman_{metric_type}",
            title=f"Spearman Corr ({metric_type.title()}) - ImageNet-C {corruption.replace('_', ' ').title()} {severity}",
            type="domain_spearman",
            x_label="",
            y_label="",
            output=f"images/spearman/{metric_type}/imagenet_c_{corruption}_{severity}.png",
            content=DefaultMunch.fromDict(content),
        )


SPEARMAN_PLOTS_TASKS = [
    CorruptionVariations(corruptions=["defocus_blur"], severities=[1]),
    CorruptionVariations(corruptions=["zoom_blur"], severities=[1]),
    CorruptionVariations(corruptions=["constrast"], severities=[1]),
]


def get_spearman_plot_specs(
    metric_type: str = "drop",
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
) -> Iterator[ChartConfig]:
    if corruptions or severities:
        space = CorruptionVariations(corruptions=corruptions, severities=severities)
        yield from construct_plot_config(space, metric_type)
    else:
        for space in SPEARMAN_PLOTS_TASKS:
            yield from construct_plot_config(space, metric_type)


def get_averaged_spearman_plot_specs(
    metric_type: str = "drop",
    groups: list[str] | None = None,
    severities: list[int] | None = None,
) -> Iterator[ChartConfig]:
    from munch import DefaultMunch

    _groups = groups or list(IMAGENET_C_CORRUPTION_GROUPS.keys())
    _severities = severities or IMAGENET_C_SEVERITIES
    _models = list(MODELS)

    for group_name in _groups:
        corruptions_in_group = IMAGENET_C_CORRUPTION_GROUPS.get(group_name)
        if not corruptions_in_group:
            continue

        models_content = []
        for model in _models:
            model_slug = model.lower().replace("-", "")
            corrupted_files = []
            for corruption in corruptions_in_group:
                for severity in _severities:
                    # Construct file name for each corruption in the group
                    corrupted_files.append(
                        f"{model_slug}_imagenet_c_{group_name}_{corruption}_{severity}.csv"
                    )

            models_content.append(
                {
                    "name": model,
                    "clean": f"{model_slug}_imagenet.csv",
                    "corrupted_files": corrupted_files,  # new field
                }
            )

        content = {
            "models": models_content,
            "metric_type": metric_type,
            "is_averaged": True,  # Flag to indicate this is an averaged plot config
        }

        severities_str = ",".join(map(str, _severities))
        yield ChartConfig(
            name=f"imagenet_c_avg_{group_name}_sev_{severities_str}_spearman_{metric_type}",
            title=f"Averaged Spearman Corr ({metric_type.title()}) - Group: {group_name.title()}, Severities: {severities_str}",
            type="domain_spearman_averaged",  # New type
            x_label="",
            y_label="",
            output=f"images/spearman/{metric_type}/imagenet_c_avg_{group_name}_sev_{severities_str}.png",
            content=DefaultMunch.fromDict(content),
        )
