from nltk.corpus import wordnet as wn
import nltk
import json


def download_wordnet_if_needed():
    """
    Downloads the WordNet corpus if it's not already downloaded.
    """
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def read_human_readable_labels() -> dict:
    with open("imagenet_class_index.json") as f:
        return {int(k): v[0] for k, v in json.load(f).items()}


def get_wordnet_similarity(synsets: list[str], indexes: list[int] = []) -> float | None:
    """
    Calculates the WordNet path similarity between two classes.
    Args:
        synsets: List of two synset name strings.
        indexes: List of two ImageNet class indexes.
    Returns:
        The path similarity score, or None if the classes are not found.
    """
    download_wordnet_if_needed()

    first_synset, second_synset = None, None

    if indexes:
        index_to_synset = read_human_readable_labels()
        first_synset, second_synset = [index_to_synset[idx] for idx in indexes]
    elif synsets:
        first_synset, second_synset = synsets

    if first_synset is None or second_synset is None:
        return None

    syn1 = wn.synset(first_synset)
    syn2 = wn.synset(second_synset)

    return syn1.path_similarity(syn2)
