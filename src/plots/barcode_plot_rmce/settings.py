from __future__ import annotations

from typing import Iterator
from plots.specs import ChartConfig
from space import CorruptionVariations


def get_barcode_plot_rmce_specs(
    models: list[str] | None = None,
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
) -> Iterator[ChartConfig]:
    from munch import DefaultMunch

    space = CorruptionVariations(
        models=models,
        corruptions=corruptions,  # FIXME
        severities=severities,
    )

    # We create one plot per corruption group
    groups = sorted(list(set(v.group for v in space)))

    for group in groups:
        content = {
            "models": models or ["resnet50"],
            "group": group,
            "corruptions": corruptions,
            "severities": severities,
        }

        yield ChartConfig(
            name=f"barcode_rmce_{group}",
            title=f"Fragile Classes (RmCE > 1.0) - {group.replace('_', ' ').title()}",
            type="barcode_rmce",
            x_label="Klasy ImageNet (0 - 999)",
            y_label="",
            output=f"images/barcode_rmce/barcode_{group}.png",
            content=DefaultMunch.fromDict(content),
        )
