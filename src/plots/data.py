from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd


logger = logging.getLogger(__name__)


def get_data(data_dir: Path | str, filename: str) -> pd.DataFrame:
    path = Path(data_dir) / filename
    return _DataLoader.load(path)


class _DataLoader:
    @staticmethod
    def exists(*paths: Path) -> bool:
        for p in paths:
            if not p.exists():
                logger.error(f"File not found: {p}")
                return False
        return True

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        logger.info(f"Loading data from {path}")
        if not _DataLoader.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        return pd.read_csv(path)
