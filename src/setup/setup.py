from __future__ import annotations

import urllib.request
from pathlib import Path
import subprocess
import tempfile
from dataset import DatasetType
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)

__all__ = ["setup_dataset"]


@dataclass(frozen=True)
class _DatasetSourceConfig:
    base_url: str
    default_archives: list[str]


DOWNLOAD_REGISTRY = {
    DatasetType.IMAGENET_C: _DatasetSourceConfig(
        base_url="https://zenodo.org/records/2235448/files",
        default_archives=[
            "noise.tar",
            "blur.tar",
            "weather.tar",
            "digital.tar",
            "extra.tar",
        ],
    ),
    DatasetType.IMAGENET_P: _DatasetSourceConfig(
        base_url="https://zenodo.org/records/3565846/files",
        default_archives=[
            "blur.tar",
            "digital.tar",
            "noise.tar",
            "weather.tar",
        ],
    ),
    DatasetType.IMAGENET_A: _DatasetSourceConfig(
        base_url="https://people.eecs.berkeley.edu/~hendrycks",
        default_archives=["imagenet-a.tar"],
    ),
}


def setup_dataset(data_root: Path, dataset: str, archives: list[str]):
    return _Downloader.setup(data_root, dataset, archives)


class _Downloader:
    @staticmethod
    def setup(data_root: Path, dataset: str, archives: list[str]):
        try:
            dataset_type = DatasetType(dataset.lower())
        except ValueError:
            logger.error(f"Unknown dataset type: {dataset}")
            return

        if dataset_type not in DOWNLOAD_REGISTRY:
            logger.warning(f"Downloading {dataset_type.value} not implemented")
            return

        source_config = DOWNLOAD_REGISTRY[dataset_type]

        _Downloader._download_dataset(data_root, dataset_type, source_config, archives)

    @staticmethod
    def _download_dataset(
        data_root: Path,
        dataset_type: DatasetType,
        source_config: _DatasetSourceConfig,
        specific_archives: list[str] = None,
    ):
        data_root = Path(data_root)

        dataset_root = data_root / dataset_type.value
        dataset_root.mkdir(parents=True, exist_ok=True)

        archives_to_download = (
            specific_archives if specific_archives else source_config.default_archives
        )

        for archive in archives_to_download:
            if not archive.endswith(".tar"):
                archive = f"{archive}.tar"

            category_name = archive.replace(".tar", "")
            category_path = dataset_root / category_name
            if category_path.exists() and any(category_path.iterdir()):
                logger.info(
                    f"Directory {category_path} already exists. Skipping download."
                )
                continue

            archive_url = f"{source_config.base_url}/{archive}"

            logger.info(f"Downloading {archive_url}...")
            with tempfile.TemporaryDirectory() as tmp:
                archive_path = Path(tmp) / archive
                try:
                    urllib.request.urlretrieve(archive_url, archive_path)
                except Exception as e:
                    logger.error(f"Error downloading {archive}: {e}")
                    continue

                logger.info(f"Extracting {archive}...")
                subprocess.run(
                    [
                        "tar",
                        "-xf",
                        str(archive_path),
                        "-C",
                        str(dataset_root),
                    ],
                    check=True,
                )
