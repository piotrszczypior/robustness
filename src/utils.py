import logging
from regex import P
import torch
import json


def resolve_device():
    logger = logging.getLogger("device")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    return device


def get_synset_to_index_imagenet1k():
    """
    Maps WordNet synset IDs to ImageNet-1K indices.

    Returns:
        dict: {"n01440764": 0, "n01443537": 1}
    """
    with open("imagenet_class_index.json", "r") as file:
        index_to_synset = json.load(file)

    return {
        synset_and_target[0]: int(index)
        for index, synset_and_target in index_to_synset.items()
    }


def get_index_to_synset_and_label_imagenet1k():
    """
    Returns:
        dict: {0: ["n01440764", "tench"], 1: ["n01443537", "goldfish"]}
    """

    with open("imagenet_class_index.json", "r") as file:
        index_to_synset = json.load(file)

    return {
        int(index): synset_and_target
        for index, synset_and_target in index_to_synset.items()
    }


def get_synset_to_label_imagenet1k():
    """
    Returns:
        dict: {n01440764: "tench"}...}
    """

    with open("imagenet_class_index.json", "r") as file:
        index_to_synset = json.load(file)

    return {
        synset_and_target[0]: synset_and_target[1]
        for _, synset_and_target in index_to_synset.items()
    }


def get_synset_to_imagenet_a_index():
    from constants import IMAGENET_A_SYNSETS

    return {synset: int(i) for i, synset in enumerate(IMAGENET_A_SYNSETS)}


def get_synset_to_imagenet_r_index():
    from constants import IMAGENET_R_SYNSETS

    return {synset: int(i) for i, synset in enumerate(IMAGENET_R_SYNSETS)}
