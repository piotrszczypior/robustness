from __future__ import annotations

import argparse
from typing import Iterator, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import yaml
import logging

from dataset import DatasetConfig, DatasetType


__all__ = ["read_experiments", "Experiment"]


logger = logging.getLogger(__name__)


def read_experiments(args: argparse.Namespace) -> list[Experiment]:
    experiments = list(_ExperimentFactory.from_yaml(args.experiments))

    logger.info(f"Found {len(experiments)} experiments in the configuration.")

    if args.run_batch is not None:
        logger.info(f"Filtering experiments by batch prefix: '{args.run_batch}'")
        experiments = [exp for exp in experiments if exp.batch_name == args.run_batch]

    if args.run_single is not None:
        logger.info(f"Filtering experiments by single run prefix: '{args.run_single}'")
        experiments = [exp for exp in experiments if exp.name == args.run_single]

    if not experiments:
        logger.warning(
            f"No experiments match the filter '{args.run_batch or args.run_single}'"
        )

    logger.info(f"Loaded {len(experiments)} experiments from the configuration.")
    return experiments


@dataclass(frozen=True)
class Experiment:
    name: str
    batch_name: str
    dataset_config: DatasetConfig


class _ExperimentFactory:
    @staticmethod
    def from_yaml(yaml_path: Union[str, Path]) -> Iterator[Experiment]:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return _ExperimentFactory._from_dict(config)

    @staticmethod
    def _from_dict(config: Dict[str, Any]) -> Iterator[Experiment]:
        experiments = config.get("experiments", [])

        for experiment in experiments:
            batch_name = experiment.get("name", "unnamed")

            dataset_type = DatasetType(experiment.get("type", "").lower())
            if dataset_type == "":
                raise ValueError("Missing dataset type")
            dataset_type = DatasetType(dataset_type)

            if dataset_type == DatasetType.IMAGENET_C:
                corruptions = experiment.get("corruptions")
                severities = experiment.get("severities")

                for corruption in corruptions:
                    for severity in severities:
                        name = f"{dataset_type.value}_{corruption}_{severity}"
                        yield Experiment(
                            name=name,
                            batch_name=batch_name,
                            dataset_config=DatasetConfig(
                                type=dataset_type,
                                corruption=corruption,
                                severity=severity,
                            ),
                        )

            elif dataset_type == DatasetType.IMAGENET_P:
                perturbations = experiment.get("perturbations")

                for pertubation in perturbations:
                    name = f"{dataset_type.value}_{pertubation}"
                    yield Experiment(
                        name=name,
                        batch_name=batch_name,
                        dataset_config=DatasetConfig(
                            type=dataset_type, perturbation=pertubation
                        ),
                    )

            else:
                yield Experiment(
                    name=dataset_type.value,
                    batch_name=batch_name,
                    dataset_config=DatasetConfig(type=dataset_type),
                )
