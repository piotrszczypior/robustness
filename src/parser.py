from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GlobalConfig:
    model_name: str
    data_path: Path
    output_path: str


def get_args_parser():
    parser = argparse.ArgumentParser(description="Robustness")

    # fmt: off
    parser.add_argument("--model-name", default="resnet152", type=str, help="model name")
    parser.add_argument("--data-path", default="data/", type=str, help="Path to data")
    parser.add_argument("--output-path", default="results/", type=str, help="Path to results")
    # fmt: on

    return parser


def get_config(args: argparse.Namespace):
    data_dir = Path(args.data_path)

    assert data_dir.is_dir(), f"Config directory '{args.config_dir}' does not exist"

    return GlobalConfig(
        model_name=args.model_name, data_path=data_dir, output_path=args.output_path
    )
