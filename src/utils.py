import logging
import torch
import json


def resolve_device():
    logger = logging.getLogger("device")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    return device


def get_synset_to_index_imagenet1k():
    with open("imagenet_class_index.json", "r") as file:
        index_to_synset = json.load(file)

    return {
        synset_and_target[0]: int(index)
        for index, synset_and_target in index_to_synset.items()
    }
