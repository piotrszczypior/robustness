from __future__ import annotations

from typing import Iterator
from plots.specs import ChartConfig
from space import VariationSpaceImageNetC


def construct_plot_config(space: VariationSpaceImageNetC, metric_type: str = "drop" ):
    from munch import DefaultMunch

    for group, corruption, severity, models in space.per_corruption():
        models_content = []
        for model in models:
            model_slug = model.lower().replace("-", "")
            models_content.append({
                "name": model,
                "clean": f"{model_slug}_imagenet.csv",
                "corrupted": f"{model_slug}_imagenet_c_{group}_{corruption}_{severity}.csv"
            })

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
            content=DefaultMunch.fromDict(content)
        )

SPEARMAN_PLOTS_TASKS = [
    VariationSpaceImageNetC(corruptions=["defocus_blur"], severities=[1]),
    VariationSpaceImageNetC(corruptions=["zoom_blur"], severities=[1]),
    VariationSpaceImageNetC(corruptions=["constrast"], severities=[1]),
]

def get_spearman_plot_specs(
    metric_type: str = "drop", 
    corruptions: list[str] | None = None, 
    severities: list[int] | None = None
) -> Iterator[ChartConfig]:
    if corruptions or severities:
        space = VariationSpaceImageNetC(corruptions=corruptions, severities=severities)
        yield from construct_plot_config(space, metric_type)
    else:
        for space in SPEARMAN_PLOTS_TASKS:
            yield from construct_plot_config(space, metric_type)

