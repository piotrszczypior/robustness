from __future__ import annotations

from typing import Iterator
from plots.specs import ChartConfig
from space import CorruptionVariations


def get_violin_plot_rmce_specs(
    models: list[str] | None = None,
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
) -> Iterator[ChartConfig]:
    from munch import DefaultMunch

    space = CorruptionVariations(
        models=models, corruptions=corruptions, severities=severities
    )

    # We want one plot showing multiple groups (on X axis)
    # and multiple models (dodged)

    # Infer groups from space
    groups = sorted(list(set(v.group for v in space)))

    content = {
        "models": models
        or [
            "resnet50"
        ],  # Default if none provided? Or maybe we should handle this in cli/space
        "groups": groups,
        "corruptions": corruptions,
        "severities": severities,
    }

    yield ChartConfig(
        name="violin_rmce",
        title="Relative mean Corruption Error (RmCE) per class",
        type="violin_rmce",
        x_label="Corruption Group",
        y_label="RmCE per class",
        output="images/violin_rmce/violin_rmce.png",
        content=DefaultMunch.fromDict(content),
    )
