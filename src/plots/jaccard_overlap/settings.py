from __future__ import annotations

from typing import Iterator
from munch import DefaultMunch
from plots.specs import ChartConfig
from space import CorruptionVariations


def construct_plot_config(
    space: CorruptionVariations, top_k: int = 50, tail: str = "worst"
):
    for group, corruption, severity, models in space.per_corruption():
        models_content = []
        for model in models:
            model_slug = model.lower().replace("-", "")
            models_content.append(
                {
                    "name": model,
                    "corrupted": f"{model_slug}_imagenet_c_{group}_{corruption}_{severity}.csv",
                }
            )

        content = {
            "models": models_content,
            "top_k": top_k,
            "tail": tail,
        }

        yield ChartConfig(
            name=f"imagenet_c_{corruption}_{severity}_jaccard_top{top_k}",
            title=f"Jaccard Overlap (Top-{top_k} Worst Classes) - ImageNet-C {corruption.replace('_', ' ').title()} {severity}",
            type="domain_jaccard",
            x_label="",
            y_label="",
            output=f"images/jaccard/top{top_k}/{tail}/imagenet_c_{corruption}_{severity}.png",
            content=DefaultMunch.fromDict(content),
        )


JACCARD_PLOTS_TASKS = [
    CorruptionVariations(corruptions=["defocus_blur"], severities=[1]),
]


def get_jaccard_plot_specs(
    top_k: int = 50,
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
    tail: str = "worst",
) -> Iterator[ChartConfig]:
    if corruptions or severities:
        space = CorruptionVariations(corruptions=corruptions, severities=severities)
        yield from construct_plot_config(space, top_k, tail)
    else:
        for space in JACCARD_PLOTS_TASKS:
            yield from construct_plot_config(space, top_k, tail)
