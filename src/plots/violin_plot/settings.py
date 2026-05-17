from __future__ import annotations

from typing import Iterator
from plots.specs import ChartConfig
from space import CorruptionVariations


def construct_violin_plot_config(
    space: CorruptionVariations, mode: str = "collage"
) -> Iterator[ChartConfig]:
    from munch import DefaultMunch

    if mode == "collage":
        # Group all models for a single corruption+severity
        for group, corruption, severity, models in space.per_corruption():
            models_content = []
            for model in models:
                model_slug = model.lower().replace("-", "")
                models_content.append(
                    {
                        "name": model,
                        "clean": f"{model_slug}_imagenet.csv",
                        "corrupted": f"{model_slug}_imagenet_c_{group}_{corruption}_{severity}.csv",
                        "corruption_label": f"{corruption.replace('_', ' ')} (sev={severity})",
                    }
                )

            content = {
                "models": models_content,
            }

            yield ChartConfig(
                name=f"violin_collage_{corruption}_{severity}",
                title=f"Per-class Accuracy Distribution - {corruption.replace('_', ' ').title()} (sev={severity})",
                type="violin",
                x_label="Per-class accuracy",
                y_label="",
                output=f"images/violin/collage/violin_{corruption}_{severity}.png",
                content=DefaultMunch.fromDict(content),
            )
    else:
        # One plot per model/corruption
        for v in space:
            model_slug = v.model.lower().replace("-", "")
            models_content = [
                {
                    "name": v.model,
                    "clean": f"{model_slug}_imagenet.csv",
                    "corrupted": f"{model_slug}_imagenet_c_{v.group}_{v.corruption}_{v.severity}.csv",
                    "corruption_label": f"{v.corruption.replace('_', ' ')} (sev={v.severity})",
                }
            ]

            content = {
                "models": models_content,
            }

            yield ChartConfig(
                name=f"violin_single_{v.model}_{v.corruption}_{v.severity}",
                title=f"Per-class Accuracy - {v.model} - {v.corruption.replace('_', ' ').title()} (sev={v.severity})",
                type="violin",
                x_label="Per-class accuracy",
                y_label="",
                output=f"images/violin/{model_slug}/{v.corruption}_{v.severity}.png",
                content=DefaultMunch.fromDict(content),
            )


def get_violin_plot_specs(
    mode: str = "collage",
    models: list[str] | None = None,
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
) -> Iterator[ChartConfig]:
    space = CorruptionVariations(
        models=models, corruptions=corruptions, severities=severities
    )
    yield from construct_violin_plot_config(space, mode)
