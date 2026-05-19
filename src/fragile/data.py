import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def get_per_class_accuracy(
    file_name: str, directory_path: Path | str = "results", agg_column="accuracy"
) -> pd.DataFrame:
    full_file_path = Path(directory_path) / file_name
    predictions_dataframe = _load_csv_dataset(full_file_path)

    return (
        predictions_dataframe.groupby(["synset", "y_true"])["is_correct"]
        .agg(**{agg_column: "mean"})
        .reset_index()
    )


def _validate_file_existence(*file_paths: Path) -> bool:
    for file_path in file_paths:
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
    return True


def _load_csv_dataset(file_path: Path) -> pd.DataFrame:
    # logger.info(f"Loading dataset from {file_path}")
    if not _validate_file_existence(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)
