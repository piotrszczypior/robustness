

from utils import get_index_to_synset_and_label_imagenet1k
from fragile.definitions import DEFINITIONS
from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES
from fragile.experiments import get_dfs_for_all_models 
from fragile.fragile import get_absolute_fragile, get_relative_drop_fragile
from space import CorruptionVariations  


MODELS = [
    "resnet50", "resnet152", "regnet_y_16gf", "resnext101_64x4d",
    "wide_resnet50_2", "wide_resnet101_2", "efficientnet_b4",
    "efficientnet_v2_m", "vit_b_16", "vit_l_16", "swin_b",
    "swin_v2_b", "maxvit_t", "convnext_base", "convnext_large",
]


GROUP_OF = {
    corruption: group
    for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items()
    for corruption in corruptions
}

AB = DEFINITIONS["ab"]



def main():
    idx_to_synset_label,  = get_index_to_synset_and_label_imagenet1k()

    for model in MODELS:
        pass