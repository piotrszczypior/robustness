from __future__ import annotations

import urllib.request
from pathlib import Path
import subprocess
import tempfile

IMAGENET_C_BASE_URL = "https://zenodo.org/records/2235448/files/"

IMAGENET_C_ARCHIVES = [
        "noise.tar",
        "blur.tar",
        "weather.tar",
        "digital.tar",
        "extra.tar"
    ]


class Downloader:
    @staticmethod
    def download_imagenet_c(data_root: Path):
        data_root = Path(data_root)
        imagenet_c_root = data_root / "imagenet_c"
        imagenet_c_root.mkdir(parents=True, exist_ok=True)

        for archive in IMAGENET_C_ARCHIVES:

            archive_url = f"{Downloader.IMAGENET_C_BASE_URL}{archive}"

            with tempfile.TemporaryDirectory() as tmp:
                archive_path = Path(tmp) / archive
                urllib.request.urlretrieve(archive_url, archive_path)

                subprocess.run(
                    [
                        "tar",
                        "-xf",
                        str(archive_path),
                        "-C",
                        str(imagenet_c_root),
                    ],
                    check=True,
                )


    # @staticmethod
    # def setup_dataset(dataset_type: str, data_root: Path):
    #     dataset_type = dataset_type.lower()
    #     if dataset_type == "imagenet":
    #         Downloader.download_imagenet_val(data_root)
    #     elif dataset_type == "imagenet_c":
    #         Downloader.download_imagenet_c(data_root)
    #     else:
    #         print(f"Dataset {dataset_type} download/setup not implemented.")
    #         print("Supported datasets: imagenet, imagenet_c")
