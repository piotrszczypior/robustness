import torch
from torchmetrics.classification import MulticlassAccuracy
from config import Config
from loader import get_imagenet_loader, get_imagenet_c_loader
from model import get_resnet50
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def evaluate_per_class(model, loader):
    model.eval()

    accuracy_per_class = MulticlassAccuracy(
        num_classes=Config.NUM_CLASSES, average=None, top_k=1
    )
    accuracy_per_class.to(DEVICE)

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        outputs = model(inputs)
        accuracy_per_class.update(outputs, targets)

    return accuracy_per_class.compute().cpu().numpy()


def violin_box_with_markers(ax, data, y_pos, label):
    data = np.asarray(data)

    parts = ax.violinplot(
        [data],
        positions=[y_pos],
        vert=False,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for b in parts["bodies"]:
        b.set_alpha(0.6)
        b.set_facecolor("#6D96D8")

    ax.boxplot(
        [data],
        positions=[y_pos],
        vert=False,
        widths=0.05,
        patch_artist=True,
        boxprops=dict(facecolor="none"),
        medianprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
    )

    mean = float(data.mean())
    p5 = float(np.percentile(data, 5))
    p10 = float(np.percentile(data, 10))

    ax.vlines(mean, y_pos - 0.12, y_pos + 0.12, linewidth=3)
    ax.vlines(p5, y_pos - 0.12, y_pos + 0.12, linewidth=3) 
    ax.vlines(p10, y_pos - 0.12, y_pos + 0.12, linewidth=3)

    std = float(data.std(ddof=0))
    ax.text(
        1.02,
        y_pos,
        f"{label}\n({mean * 100:.2f} ± {std * 100:.2f})",
        va="center",
        transform=ax.get_yaxis_transform(),
    )

    return mean, std, p5, p10


def test():
    model = get_resnet50()
    model.to(DEVICE)
    imagenet = get_imagenet_loader()
    imagenet_c = get_imagenet_c_loader()

    acc_per_class_clean = evaluate_per_class(model, loader=imagenet)
    acc_per_class_imagenet_c = evaluate_per_class(model, loader=imagenet_c)

    _, ax = plt.subplots(figsize=(9, 4))
    violin_box_with_markers(
        ax, acc_per_class_clean, y_pos=2, label="ResNet-50 (ImageNet-1K)"
    )
    violin_box_with_markers(
        ax,
        acc_per_class_imagenet_c,
        y_pos=1,
        label="ResNet-50 (ImageNet-C defocus_blur, sev=3)",
    )

    ax.set_xlim(0, 1)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["ImageNet-C", "ImageNet"])
    ax.set_xlabel("Per-class accuracy")
    ax.set_title("Per-class accuracy distribution")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test()
    # print(DEVICE)
