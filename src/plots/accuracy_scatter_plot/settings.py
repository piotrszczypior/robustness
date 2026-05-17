from __future__ import annotations

from plots.specs import ChartConfig
from munch import DefaultMunch


def get_accuracy_scatter_plot_specs(x_file: str, y_file: str, mode: str) -> ChartConfig:
    plot_type = (
        "accuracy_to_accuracy_drop" if mode == "drop" else "accuracy_to_accuracy"
    )

    content = {
        "x": x_file,
        "y": y_file,
    }

    x_name = x_file.replace(".csv", "").replace("/", "_")
    y_name = y_file.replace(".csv", "").replace("/", "_")

    return ChartConfig(
        name=f"accuracy_scatter_{x_name}_vs_{y_name}_{mode}",
        title="Accuracy vs. Accuracy Drop",
        type=plot_type,
        x_label="Accuracy (Clean ImageNet",
        y_label="Accuracy (Model 2)" if mode == "default" else "Accuracy Drop",
        output=f"images/accuracy_scatter/{x_name}_vs_{y_name}_{mode}.png",
        content=DefaultMunch.fromDict(content),
    )
