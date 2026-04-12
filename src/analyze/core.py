from __future__ import annotations

import importlib
from analyze.settings import BaseAnalysisConfig


def run_analysis(config: BaseAnalysisConfig, output_dir: str):
    try:
        module = importlib.import_module(f"analyze.{config.type}")
        module.run(config, output_dir)
    except ImportError:
        raise ValueError(f"Unsupported analysis type: {config.type}")
