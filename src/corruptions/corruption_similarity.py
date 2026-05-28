
from constants import IMAGENET_C_CORRUPTION_GROUPS
from model import MODELS
from fragile.experiments import get_dfs_per_setting_for_all_models
from space import CorruptionVariations
import pandas as pd 
import numpy as np
from scipy.stats import spearmanr


def find_similarities():
    model_dfs = _generate_matrix()

    acc_matrix = _build_accuracy_matrix(model_dfs)
    sim_df = _compute_spearman_similarity(acc_matrix)
    pairs = _get_cross_corruption_pairs(sim_df, threshold=0.0)

    print()
    print()

    print(pairs)

    return sim_df, pairs


def _generate_matrix():
    dfs_per_model = get_dfs_per_setting_for_all_models(CorruptionVariations(models=['resnet50']))

    return dfs_per_model['resnet50']


def _build_accuracy_matrix(model_dfs: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for (corruption, severity), df in model_dfs.items():
        label = f"{corruption}_{severity}"
        rows[label] = df.set_index("synset")["acc_corrupt"]

    return pd.DataFrame(rows).T

def _compute_spearman_similarity(acc_matrix: pd.DataFrame) -> pd.DataFrame:
    labels = acc_matrix.index.tolist()
    n = len(labels)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r, _ = spearmanr(acc_matrix.iloc[i], acc_matrix.iloc[j])
            sim[i, j] = r

    return pd.DataFrame(sim, index=labels, columns=labels)



def _get_cross_corruption_pairs(sim_df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    labels = sim_df.index.tolist()
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            corruption_i = "_".join(labels[i].split("_")[:-1])
            corruption_j = "_".join(labels[j].split("_")[:-1])
            if corruption_i == corruption_j:
                continue
            pairs.append({
                "setting_a": labels[i],
                "setting_b": labels[j],
                "spearman_r": sim_df.iloc[i, j],
            })

    df = pd.DataFrame(pairs).sort_values("spearman_r", ascending=False)
    return df[df["spearman_r"] >= threshold]
