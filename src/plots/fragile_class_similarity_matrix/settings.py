from __future__ import annotations

from plots.specs import ChartConfig


def get_fragile_class_similarity_matrix_specs(
    files: list[str], names: list[str]
) -> ChartConfig:
    plot_type = "fragile_class_similarity_matrix"

    content = []
    for name, data in zip(names, files):
        content.append({"name": name, "data": data})

    return ChartConfig(
        name=f"fragile_class_similarity_matrix_{'_'.join(names)}",
        title="Fragile Class Similarity Matrix",
        type=plot_type,
        x_label="",
        y_label="",
        output=f"images/fragile_class_similarity_matrix/{'_'.join(names)}.png",
        content=content,
    )
