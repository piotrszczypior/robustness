from __future__ import annotations

import argparse
from typing import Iterator, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import yaml
import logging
import itertools

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
    filename_suffix: str
    batch_name: str
    dataset_config: DatasetConfig


class _ExperimentFactory:
    @staticmethod
    def from_yaml(yaml_path: Union[str, Path]) -> Iterator[Experiment]:
        with open(yaml_path, "r") as f:
            contents = yaml.safe_load(f)
        return _ExperimentFactory._from_dict(contents)

    @classmethod
    def _from_dict(cls, contents: Dict[str, Any]) -> Iterator[Experiment]:
        experiments = contents.get("experiments", [])

        for experiment in experiments:
            yield from cls._parse_experiment(experiment)

    @classmethod
    def _parse_experiment(cls, experiment: Dict[str, Any]) -> Iterator[Experiment]:
        batch_name = experiment.get("name", "unnamed")

        type = experiment.get("type")
        if not type:
            raise ValueError(
                f"Missing dataset type in experiment batch: '{batch_name}'"
            )

        try:
            dataset_type = DatasetType(type.lower())
        except ValueError:
            logger.error(f"[ERROR] Unknown dataset type: {type}")
            raise ValueError(f"Unknown dataset type: {type}")

        registry = {
            DatasetType.IMAGENET_C: cls._parse_imagenet_c,
            DatasetType.IMAGENET_P: cls._parse_imagenet_p,
        }

        if dataset_type in registry:
            yield from registry[dataset_type](batch_name, dataset_type, experiment)
        else:
            yield cls._parse_default(batch_name, dataset_type)

    @staticmethod
    def _parse_imagenet_c(
        batch_name: str, dataset_type: DatasetType, experiment: Dict[str, Any]
    ) -> Iterator[Experiment]:
        corruptions = experiment.get("corruptions", [])
        severities = experiment.get("severities", [])

        for corruption, severity in itertools.product(corruptions, severities):
            yield Experiment(
                name=f"{dataset_type.value}_{corruption}_{severity}",
                filename_suffix=f"{batch_name}_{corruption}_{severity}",
                batch_name=batch_name,
                dataset_config=DatasetConfig(
                    type=dataset_type,
                    corruption=corruption,
                    severity=severity,
                ),
            )

    @staticmethod
    def _parse_imagenet_p(
        batch_name: str, dataset_type: DatasetType, experiment: Dict[str, Any]
    ) -> Iterator[Experiment]:
        perturbations = experiment.get("perturbations", [])

        for perturbation in perturbations:
            yield Experiment(
                name=f"{dataset_type.value}_{perturbation}",
                filename_suffix=f"{dataset_type.value}_{perturbation}",
                batch_name=batch_name,
                dataset_config=DatasetConfig(
                    type=dataset_type, perturbation=perturbation
                ),
            )

    @staticmethod
    def _parse_default(batch_name: str, dataset_type: DatasetType) -> Experiment:
        return Experiment(
            name=dataset_type.value,
            filename_suffix=dataset_type.value,
            batch_name=batch_name,
            dataset_config=DatasetConfig(type=dataset_type),
        )
