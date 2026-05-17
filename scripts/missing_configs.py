import os

# Konfiguracja ścieżek
RESULTS_DIR = "results"
OUTPUT_FILE = "missing_configs_grouped.txt"

# Słowniki z definicjami
IMAGENET_C_CORRUPTION_GROUPS = {
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "digital": ["contrast", "elastic_transform", "jpeg_compression", "pixelate"],
    "noise": ["gaussian_noise", "impulse_noise", "shot_noise"],
    "weather": ["brightness", "fog", "frost", "snow"],
    # "extra": ["gaussian_blur", "saturate", "spatter", "speckle_noise"],
}

MODELS = {
    "alexnet": "AlexNet",
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "resnet152": "ResNet-152",
    "regnet_y_16gf": "RegNet-Y-16GF",
    "resnext101_64x4d": "ResNeXt-101-64x4d",
    "wide_resnet50_2": "Wide-ResNet-50-2",
    "wide_resnet101_2": "Wide-ResNet-101-2",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b4": "EfficientNet-B4",
    "efficientnet_v2_m": "EfficientNet-V2-M",
    "densenet121": "DenseNet-121",
    "mobilenet_v3_large": "MobileNet-V3-Large",
    "vit_b_16": "ViT-B/16",
    "vit_l_16": "ViT-L/16",
    "swin_b": "Swin-B",
    "swin_v2_b": "Swin-V2-B",
    "maxvit_t": "MaxVit-T",
    "convnext_base": "ConvNeXt-Base",
    "convnext_large": "ConvNeXt-Large",
}

STANDARD_DATASETS = ["imagenet", "imagenet_a", "imagenet_r"]


def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Błąd: Katalog '{RESULTS_DIR}' nie istnieje w obecnej ścieżce.")
        existing_files = set()
    else:
        existing_files = set(os.listdir(RESULTS_DIR))

    missing_grouped_by_model = {}

    for model in MODELS.keys():
        missing_for_model = []

        # 1. Sprawdzanie standardowych datasetów
        for dataset in STANDARD_DATASETS:
            filename = f"{model}_{dataset}.csv"
            if filename not in existing_files:
                missing_for_model.append(dataset)

        # 2. Sprawdzanie ImageNet-C
        for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items():
            for corruption in corruptions:
                missing_severities = []

                # Zliczamy braki dla konkretnej korupcji
                for severity in range(1, 6):
                    filename = f"{model}_imagenet_c_{group}_{corruption}_{severity}.csv"
                    if filename not in existing_files:
                        missing_severities.append(severity)

                # Logika grupowania
                if len(missing_severities) >= 4:
                    # Brakuje 4 lub 5 plików -> wrzucamy samą nazwę korupcji
                    missing_for_model.append(f"{group}_{corruption}")
                elif len(missing_severities) > 0:
                    # Brakuje 1 do 3 plików -> wypisujemy konkretne severity
                    for s in missing_severities:
                        missing_for_model.append(f"{group}_{corruption}_{s}")

        # Zapisujemy tylko jeśli model ma jakiekolwiek braki
        if missing_for_model:
            missing_grouped_by_model[model] = missing_for_model

    # Zapis wyników do pliku
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for model, missing_items in missing_grouped_by_model.items():
            # Łączymy listę braków przecinkami
            joined_items = ", ".join(missing_items)
            f.write(f"{model} - {joined_items}\n")

    print(f"Gotowe! Znaleziono braki dla {len(missing_grouped_by_model)} modeli.")
    print(f"Pogrupowana lista została zapisana w pliku: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
