from __future__ import annotations

from typing import Iterator
from plots.specs import ChartConfig
from space import CorruptionVariations


def get_barcode_fragile_classes_specs(
    models: list[str] | None = None,
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
    group: list[int] | None = None,
) -> Iterator[ChartConfig]:
    from munch import DefaultMunch

    space = CorruptionVariations(
        models=models, corruptions=corruptions, severities=severities, groups=group
    )

    groups = sorted(list(set(v.group for v in space)))

    for group in groups:
        content = {
            "models": models or [],
            "group": group,
            "corruptions": corruptions,
            "severities": severities,
        }

        corr_suffix = "_" + ",".join(sorted(corruptions)) if corruptions else ""
        sev_suffix = (
            "_sev_" + ",".join(str(s) for s in sorted(severities)) if severities else ""
        )
        file_stem = f"barcode_{group}{corr_suffix}{sev_suffix}"

        yield ChartConfig(
            name=f"barcode_fragile_classes_{group}{corr_suffix}{sev_suffix}",
            title=f"Fragile Classes - {group.replace('_', ' ').title()}",
            type="barcode_fragile_classes",
            x_label="ImageNet classes",
            y_label="",
            output=f"images/barcode_fragile_classes/{file_stem}.png",
            content=DefaultMunch.fromDict(content),
        )
