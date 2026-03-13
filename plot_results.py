import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

matplotlib.use("TkAgg")


def plot_accuracies(csv_path):
    # Read the data
    df = pd.read_csv(csv_path)
    accuracies = df["accuracy"].values

    # Set the style
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Histogram
    sns.histplot(accuracies, bins=50, kde=True, ax=ax1, color="skyblue")
    ax1.set_title("Distribution of Per-Class Accuracies")
    ax1.set_xlabel("Accuracy")
    ax1.set_ylabel("Frequency")
    ax1.axvline(
        accuracies.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {accuracies.mean():.4f}",
    )
    ax1.legend()

    # 2. Sorted plot (to see the "tail")
    sorted_acc = np.sort(accuracies)
    ax2.plot(range(len(sorted_acc)), sorted_acc, color="blue", alpha=0.6)
    ax2.fill_between(range(len(sorted_acc)), sorted_acc, color="blue", alpha=0.2)
    ax2.set_title("Sorted Per-Class Accuracies")
    ax2.set_xlabel("Class Index (Sorted)")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    output_img = csv_path.replace(".csv", "_plot.png")
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")

    # Summary stats
    print(f"Mean: {accuracies.mean():.4f}")
    print(f"Median: {np.median(accuracies):.4f}")
    print(
        f"Min: {accuracies.min():.4f} (Class {df.loc[df['accuracy'].idxmin(), 'class_id']})"
    )
    print(
        f"Max: {accuracies.max():.4f} (Class {df.loc[df['accuracy'].idxmax(), 'class_id']})"
    )


if __name__ == "__main__":
    import sys

    path = "results/per_class_accuracy_resnet152.csv"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    plot_accuracies(path)
