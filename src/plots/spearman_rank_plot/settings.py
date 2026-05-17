from __future__ import annotations

from plots.specs import ChartConfig


def get_spearman_rank_plot_specs(files: list[str], names: list[str]) -> ChartConfig:
    plot_type = "spearman_rank_plot"

    content = []
    for name, data in zip(names, files):
        content.append({"name": name, "data": data})

    return ChartConfig(
        name=f"spearman_rank_{'_'.join(names)}",
        title="Spearman Rank Correlation Plot",
        type=plot_type,
        x_label="",
        y_label="",
        output=f"images/spearman_rank/{'_'.join(names)}.png",
        content=content,
    )
