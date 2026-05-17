from __future__ import annotations

from plots.specs import ChartConfig
from munch import DefaultMunch


def get_class_degradation_plot_specs(
    baseline_label: str, baseline_data: str, degraded_label: str, degraded_data: str
) -> ChartConfig:
    content = {
        "baseline": {"label": baseline_label, "data": baseline_data},
        "degraded": {"label": degraded_label, "data": degraded_data},
    }

    baseline_name = baseline_data.replace(".csv", "").replace("/", "_")
    degraded_name = degraded_data.replace(".csv", "").replace("/", "_")

    return ChartConfig(
        name=f"class_degradation_{baseline_name}_vs_{degraded_name}",
        title=f"Class Degradation Plot: {baseline_label} vs {degraded_label}",
        type="sorted_index_class_degradation",
        x_label="Class Index (Sorted by Baseline Accuracy)",
        y_label="Accuracy",
        output=f"images/class_degradation/{baseline_name}_vs_{degraded_name}.png",
        content=DefaultMunch.fromDict(content),
    )
