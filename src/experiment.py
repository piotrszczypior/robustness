from __future__ import annotations

from typing import List, Iterator, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import yaml

from dataset import DatasetConfig, DatasetType


__all__ = ["read_experiments", "Experiment"]


import logging

logger = logging.getLogger(__name__)


def read_experiments(config) -> list[Experiment]:
    experiments = list(ExperimentFactory.from_yaml(config.experiments_path))

    if config.run is None:
        logger.info(f"Loaded {len(experiments)} experiments from the configuration.")
        return experiments

    logger.info(f"Filtering experiments by prefix: '{config.run}'")

    run_experiments = [exp for exp in experiments if exp.name.startswith(config.run)]

    if not run_experiments:
        logger.warning(f"No experiments match the filter '{config.run}'")

    logger.info(f"Loaded {len(run_experiments)} experiments from the configuration.")
    return run_experiments


@dataclass(frozen=True)
class Experiment:
    name: str
    dataset_config: DatasetConfig


class ExperimentFactory:
    @staticmethod
    def from_yaml(yaml_path: Union[str, Path]) -> Iterator[Experiment]:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return ExperimentFactory._from_dict(config)

    @staticmethod
    def _from_dict(config: Dict[str, Any]) -> Iterator[Experiment]:
        experiments = config.get("experiments", [])

        for experiment in experiments:
            dataset_name = experiment.get("name", "unnamed")

            dataset_type = DatasetType(experiment.get("type", "").lower())
            if dataset_type == "":
                raise Exception()
            dataset_type = DatasetType(dataset_type)

            if dataset_type == DatasetType.IMAGENET_C:
                corruptions = experiment.get("corruptions")
                severities = experiment.get("severities")

                for corruption in corruptions:
                    for severity in severities:
                        name = f"{dataset_name}_{corruption}_{severity}"
                        yield Experiment(
                            name=name,
                            dataset_config=DatasetConfig(
                                type=dataset_type,
                                corruption=corruption,
                                severity=severity,
                            ),
                        )

            elif dataset_type == DatasetType.IMAGENET_P:
                perturbations = experiment.get("perturbations")

                for pertubation in perturbations:
                    name = f"{dataset_name}_{pertubation}"
                    yield Experiment(
                        name=name,
                        dataset_config=DatasetConfig(
                            type=dataset_type, perturbation=pertubation
                        ),
                    )

            else:
                yield Experiment(
                    name=dataset_type.value,
                    dataset_config=DatasetConfig(type=dataset_type),
                )
