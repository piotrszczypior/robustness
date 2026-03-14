from __future__ import annotations

import os
import logging
import torch

from parser import GlobalConfig, get_args_parser, get_config
from evaluate import run_evaluation
from model import get_model
from config import Config
from experiment import read_experiments


def setup_logging():
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(Config.LOGS_DIR, Config.LOG_FILE)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def main(config: GlobalConfig) -> int:
    logger = setup_logging()
    logger.info(f"Starting evaluation of {config.model_name}")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Output path: {config.output_path}")

    model = get_model(config.model_name)
    experiments = read_experiments(config)
    logger.info(f"Found {len(experiments)} experiments")

    run_evaluation(
        config=config,
        model=model,
        experiments=experiments,
    )

    logger.info("Evaluation finished.")
    return 0


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    config = get_config(args=args)
    main(config)
