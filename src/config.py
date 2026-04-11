from __future__ import annotations

__all__ = ["Config"]


class Config:
    DATA_ROOT = "data/"
    RESULTS_DIR = "results2/"
    LOGS_DIR = "logs/"
    LOG_FILE = "run.log"
    BATCH_SIZE = 128
    NUM_WORKERS = 2
    NUM_CLASSES = 1000
    GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/robustness/results2"
