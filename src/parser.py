from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dataset import DatasetType


DEFAULT_MODEL_NAME = "resnet152"
DEFAULT_DATAPATH = "data/"
DEFAULT_OUTPUT_PATH = "results/"
DEFAULT_DATASET = DatasetType.IMAGENET.value


@dataclass(frozen=True)
class GlobalConfig:
    model_name: str
    data_path: Path
    experiments_path: str
    output_path: str


def get_args_parser():
    parser = argparse.ArgumentParser(description="Robustness")
    # subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # fmt: off
    parser.add_argument("--model-name", default="resnet152", type=str, help="model name")
    parser.add_argument("--data-path", default="data/", type=str, help="Path to data")
    parser.add_argument("--output-path", default="results/", type=str, help="Path to results")
    parser.add_argument("--experiments-path", type=str, help="Experiments")
    # fmt: on

    return parser


def get_config(args: argparse.Namespace):
    data_path = Path(args.data_path)

    # FIXME: download mode
    # if args.command == "download":
    # data_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory '{data_path}' does not exist")

    # FIXME: manual mode
    return GlobalConfig(
        model_name=args.model_name or DEFAULT_MODEL_NAME,
        data_path=data_path,
        experiments_path=args.experiments_path,
        output_path=args.output_path or DEFAULT_OUTPUT_PATH,
    )
