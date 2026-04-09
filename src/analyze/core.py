from __future__ import annotations

from analyze.specs import AnalysisConfig

from .tasks import ClassDegradationAnalysis


def run_analysis(config: AnalysisConfig, output_dir: str):
    if config.type == "class_degradation":
        task = ClassDegradationAnalysis(config, output_dir)
        task.run()
        return

    raise ValueError(f"Unsupported analysis type: {config.type}")
