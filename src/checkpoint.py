from __future__ import annotations

from config import Config
import pandas as pd
import shutil
import os
import json

__all__ = ["MetricsExporter"]


class MetricsExporter:
    def __init__(self, output_dir: str = Config.RESULTS_DIR, backup_dir: str = None):
        self.output_dir = output_dir
        self.backup_dir = backup_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if self.backup_dir:
            os.makedirs(self.backup_dir, exist_ok=True)

    def export(self, data_df: pd.DataFrame, metadata: dict, filename: str) -> str:
        if not filename.endswith(".json"):
            filename += ".json"
        path = os.path.join(self.output_dir, filename)

        payload = {
            "metadata": metadata,
            "results": data_df.to_dict(orient="records"),
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        if self.backup_dir:
            shutil.copy2(path, self.backup_dir)
        return path
