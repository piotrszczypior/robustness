from __future__ import annotations

from config import Config
import pandas as pd
import shutil
import os

__all__ = ["MetricsExporter"]


class MetricsExporter:
    def __init__(self, output_dir: str = Config.RESULTS_DIR, backup_dir: str = None):
        self.output_dir = output_dir
        self.backup_dir = backup_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if self.backup_dir:
            os.makedirs(self.backup_dir, exist_ok=True)

    def export(self, data_df: pd.DataFrame, filename: str):
        if not filename.endswith(".csv"):
            filename += ".csv"

        file_path = os.path.join(self.output_dir, filename)
        data_df.to_csv(file_path, index=False)

        if self.backup_dir:
            self._backup_results(file_path)

        return file_path

    def _backup_results(self, file_path):
        shutil.copy2(file_path, self.backup_dir)
