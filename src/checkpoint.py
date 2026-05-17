from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from paths import paths

__all__ = ["export_results"]


def export_results(
    data_df: pd.DataFrame,
    filename: str,
    output_dir: str | Path | None = None,
    backup_dir: str | Path | None = None,
) -> Path:
    output_dir = Path(output_dir) if output_dir else paths.results
    output_dir.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".csv"):
        filename += ".csv"

    file_path = output_dir / filename
    data_df.to_csv(file_path, index=False)

    if backup_dir:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_dir)

    return file_path
