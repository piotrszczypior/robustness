import logging
import torch


def resolve_device():
    logger = logging.getLogger("device")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    return device
