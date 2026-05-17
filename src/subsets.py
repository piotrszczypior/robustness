import json
from nltk.corpus import wordnet as wn
from analyze.wordnet import download_wordnet_if_needed
from utils import get_synset_to_label_imagenet1k
import pandas as pd
from utils import get_synset_to_index_imagenet1k
from constants import IMAGENET_R_SYNSETS


def filder_imagenet_r_classes(df: pd.DataFrame):
    synset_to_index = get_synset_to_index_imagenet1k()
    imagenet_r_indices = {
        synset_to_index[s] for s in IMAGENET_R_SYNSETS if s in synset_to_index
    }

    return df[df["y_true"].isin(imagenet_r_indices)]


def group_imagenet_r_classes():
    """
    Groups ImageNet-R classes into high-level categories.
    """
    download_wordnet_if_needed()

    HIGH_LEVEL_CATEGORIES = {
        "animal": wn.synset("animal.n.01"),
        "vehicle": wn.synset("vehicle.n.01"),
        "food": wn.synset("food.n.01"),
        "object": wn.synset(
            "artifact.n.01"
        ),  # 'artifact' is a good proxy for a general 'object' category
    }

    class_index = get_synset_to_label_imagenet1k()

    grouped_classes = {name: [] for name in HIGH_LEVEL_CATEGORIES.keys()}
    grouped_classes["other"] = []

    for wnid in IMAGENET_R_SYNSETS:
        synset = wn.synset(wnid)
        hypernym_paths = synset.hypernym_paths()

        assigned_category = None

        for category_name, category_synset in HIGH_LEVEL_CATEGORIES.items():
            # Check if the category synset is in any of the hypernym paths
            if any(category_synset in path for path in hypernym_paths):
                assigned_category = category_name
                break

        class_info = {"wnid": wnid, "name": class_index.get(wnid, "Unknown")}

        if assigned_category:
            grouped_classes[assigned_category].append(class_info)
        else:
            grouped_classes["other"].append(class_info)

    return grouped_classes


if __name__ == "__main__":
    grouped = group_imagenet_r_classes()
    print(json.dumps(grouped, indent=2))
