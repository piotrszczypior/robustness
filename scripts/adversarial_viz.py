from __future__ import annotations

import argparse
import sys
from pathlib import Path
from adjustText import adjust_text

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

_COLOR_FRAGILE = "#C0392B"
_COLOR_NON_FRAGILE = "#2980B9"

_EPSILON_LABELS = ["1/255", "2/255", "4/255", "8/255"]
_EPSILON_4 = 4 / 255


def _style(ax: "plt.Axes") -> None:
    """Apply adversarial_dot house style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=1, zorder=0)
    ax.tick_params(left=False)


def _round_eps(v: float) -> float:
    return round(v, 8)


def _load_adversarial(adv_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(adv_dir.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No CSVs in {adv_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["_eps"] = df["epsilon"].apply(_round_eps)
    return df


def _load_corruption_acc_drop(results_dir: Path, model: str, synsets: list[str]) -> pd.Series:
    """Mean acc_drop per synset across all ImageNet-C conditions."""
    clean_path = results_dir / f"{model}_imagenet.csv"
    clean_df = pd.read_csv(clean_path)
    acc_clean = (
        clean_df[clean_df["synset"].isin(synsets)]
        .groupby("synset")["is_correct"]
        .mean()
    )

    drops: dict[str, list[float]] = {s: [] for s in synsets}
    for csv in sorted(results_dir.glob(f"{model}_imagenet_c_*.csv")):
        cdf = pd.read_csv(csv)
        cdf = cdf[cdf["synset"].isin(synsets)]
        if cdf.empty:
            continue
        acc_c = cdf.groupby("synset")["is_correct"].mean()
        for synset in synsets:
            if synset in acc_clean.index and synset in acc_c.index:
                drops[synset].append(float(acc_clean[synset] - acc_c[synset]))

    return pd.Series(
        {s: float(np.mean(v)) if v else float("nan") for s, v in drops.items()},
        name="corruption_acc_drop",
    )


def plot_line(df: pd.DataFrame, attack: str, out_dir: Path) -> None:
    sub = df[df["attack"] == attack].copy()
    eps_vals = sorted(sub["_eps"].unique(), key=lambda x: round(x, 8))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    for is_frag, label, color in [(1, "Fragile", _COLOR_FRAGILE), (0, "Robust", _COLOR_NON_FRAGILE)]:
        grp = sub[sub["is_fragile"] == is_frag]

        means, stds = [], []
        for eps in eps_vals:
            rows = grp[grp["_eps"] == _round_eps(eps)]
            means.append(rows["adv_acc"].mean())
            stds.append(rows["adv_acc"].std())

        means = np.array(means)
        stds = np.array(stds)
        x = np.arange(len(eps_vals))

        ax.plot(x, means, color=color, linewidth=2, marker="o", markersize=6, label=label, zorder=3)
        ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.15, zorder=2)

    ax.set_xticks(np.arange(len(eps_vals)))
    ax.set_xticklabels(_EPSILON_LABELS, fontsize=12)
    ax.set_xlabel("ε", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0.0, 1.1, 0.1)], fontsize=12)
    ax.set_title(attack.upper(), fontsize=14, pad=6)
    _style(ax)
    ax.legend(fontsize=11)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = out_dir / f"adversarial_line_{attack}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, corruption_drop: pd.Series, out_dir: Path) -> None:
    eps4 = _round_eps(_EPSILON_4)
    sub = df[df["_eps"] == eps4].copy()

    for attack in ["fgsm", "pgd"]:
        att = sub[sub["attack"] == attack].copy()

        att = att.set_index("synset")
        common = att.index.intersection(corruption_drop.dropna().index)
        att = att.loc[common]
        corr = corruption_drop.loc[common]

        colors = [_COLOR_FRAGILE if f == 1 else _COLOR_NON_FRAGILE for f in att["is_fragile"]]

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("white")

        ax.scatter(corr, att["acc_drop"], c=colors, s=50, alpha=0.8, zorder=3)

        top5 = att["acc_drop"].nlargest(10).index
        texts = []
        for syn in top5:
            if syn in corr.index:
                name = att.loc[syn, "class_name"]
                t = ax.text(
                    corr[syn], att.loc[syn, "acc_drop"],
                    name, fontsize=8,
                )
                texts.append(t)

        adjust_text(
            texts,
            x=corr.values,
            y=att["acc_drop"].values,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8),
            expand=(2.5, 3.0),
            force_text=(1.0, 2.0),
            force_points=(0.5, 1.0),
            iter_lim=700,
        )

        rho, pval = stats.spearmanr(corr, att["acc_drop"])
        ax.text(
            0.97, 0.05,
            f"ρ = {rho:.2f}\np = {pval:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
        )

        ax.set_xlabel("Mean acc drop — corruption", fontsize=14)
        ax.set_ylabel("Adversarial accuracy drop  (ε=4/255)", fontsize=14)
        ax.set_title(attack.upper(), fontsize=14, pad=6)
        _style(ax)

        ax.spines["left"].set_visible(True)
        ax.tick_params(left=True, labelsize=11)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            out = out_dir / f"adversarial_scatter_{attack}.{ext}"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"Saved: {out}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial robustness visualizations")
    parser.add_argument("--adv-dir", default="aversarial", help="Directory with adversarial CSVs")
    parser.add_argument("--results-dir", default="results", help="Directory with ImageNet-C result CSVs")
    parser.add_argument("--model", default="vit_b_16")
    parser.add_argument("--output-dir", default="images/adversarial/viz")
    args = parser.parse_args()

    adv_dir = Path(args.adv_dir)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading adversarial results...", file=sys.stderr)
    df = _load_adversarial(adv_dir)

    for attack in ["fgsm", "pgd"]:
        if attack in df["attack"].unique():
            plot_line(df, attack, out_dir)

    print("Loading corruption drop", file=sys.stderr)
    synsets = df["synset"].unique().tolist()
    corruption_drop = _load_corruption_acc_drop(results_dir, args.model, synsets)

    plot_scatter(df, corruption_drop, out_dir)


if __name__ == "__main__":
    main()
