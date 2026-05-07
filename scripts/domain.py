import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. KONFIGURACJA
# ==========================================
RESULTS_DIR = "results"
BASE_DOMAIN = "imagenet"  # suffix dla czystego zbioru
TARGET_DOMAIN = "imagenet_c"  # suffix dla domeny OOD

# Słownik mapujący nazwy modeli (do wykresu) na prefixy plików (z Twojego folderu)
# Uzupełnij tę listę zgodnie ze swoimi plikami!
MODELS_MAP = {
    "AlexNet": "alexnet",
    "ResNet-50": "resnet50",
    "ResNet-152": "resnet152",
    "DenseNet-121": "densenet121",
    "EfficientNet-B4": "efficientnet_b4",
    "ConvNeXt-Base": "convnext_base",
    "ViT-B/16": "vit_b_16",
    "ViT-L/16": "vit_l_16",
    # "ViT-H/14 (SWAG)": "vit_h_14",
    # "ViT-L/16 (I-JEPA)": "vit_l_16_jepa",
    # "ViT-H/14 (I-JEPA)": "vit_h_14_jepa"
    # ... dodaj resztę z tabeli
}


# ==========================================
# 2. FUNKCJE POMOCNICZE
# ==========================================
def get_class_accuracy_from_csv(file_path):
    """Zwraca Pandas Series z dokładnością (accuracy) dla każdej klasy."""

    # Ładujemy tylko dwie kolumny, ignorując całą resztę (ogromna oszczędność RAM)
    try:
        df = pd.read_csv(file_path, usecols=['y_true', 'is_correct'])
    except ValueError as e:
        print(f"Błąd kolumn w pliku {file_path}. Zignorowano. Szczegóły: {e}")
        return None

    # Upewniamy się, że is_correct to wartości numeryczne (0.0 / 1.0),
    # na wypadek gdyby w pliku zapisały się jako stringi 'True'/'False'
    df['is_correct'] = df['is_correct'].astype(float)

    # Grupujemy po prawdziwej klasie i wyciągamy średnią (co daje nam ułamek poprawnych)
    return df.groupby('y_true')['is_correct'].mean()


def load_model_domain_drop(model_name, prefix):
    """Oblicza wektor Drop (spadek dokładności) per klasa dla danego modelu."""
    print(f"Przetwarzanie modelu: {model_name} ({prefix})...")

    # 1. Wczytaj bazowy ImageNet (czysty)
    base_file = os.path.join(RESULTS_DIR, f"{prefix}_{BASE_DOMAIN}.csv")
    if not os.path.exists(base_file):
        print(f"  [UWAGA] Brak pliku bazowego: {base_file}")
        return None

    acc_clean = get_class_accuracy_from_csv(base_file)

    # 2. Znajdź WSZYSTKIE pliki dla domeny docelowej (np. wszystkie szumy i severity dla IN-C)
    search_pattern = os.path.join(RESULTS_DIR, f"{prefix}_{TARGET_DOMAIN}_*.csv")
    target_files = glob.glob(search_pattern)

    # Jeśli to np. ImageNet-A/R (jeden plik), sprawdź też dokładne dopasowanie bez '_' na końcu
    if not target_files:
        exact_match = os.path.join(RESULTS_DIR, f"{prefix}_{TARGET_DOMAIN}.csv")
        if os.path.exists(exact_match):
            target_files = [exact_match]

    if not target_files:
        print(f"  [UWAGA] Brak plików domeny OOD dla patternu: {search_pattern}")
        return None

    # 3. Wczytaj i uśrednij wyniki ze wszystkich zniekształceń
    # Dzięki temu mamy uśrednioną odporność na CAŁE środowisko ImageNet-C
    acc_targets = []
    for f in target_files:
        acc_targets.append(get_class_accuracy_from_csv(f))

    # Tworzymy DataFrame ze wszystkich plików i wyciągamy średnią (mean) po wierszach
    acc_target_mean = pd.concat(acc_targets, axis=1).mean(axis=1)

    # 4. Obliczamy Absolute Drop (lub Relative Drop, jeśli wolisz)
    # Różnica: czysty_wynik - zepsuty_wynik
    drop_per_class = acc_clean - acc_target_mean

    # Usuwamy NaN (jeśli domena ma tylko 200 klas jak ImageNet-R, tutaj to odfiltrujemy!)
    drop_per_class = drop_per_class.dropna()
    drop_per_class.name = model_name

    return drop_per_class


# ==========================================
# 3. GŁÓWNA LOGIKA
# ==========================================
def main():
    drops_list = []

    # Zbieramy dane dla wszystkich zadeklarowanych modeli
    for model_name, prefix in MODELS_MAP.items():
        drop_series = load_model_domain_drop(model_name, prefix)
        if drop_series is not None:
            drops_list.append(drop_series)

    if not drops_list:
        print("Nie udało się załadować żadnych danych. Sprawdź ścieżki i nazwy plików.")
        return

    # Łączymy wszystkie serie w jedną dużą tabelę (Wiersze: klasy, Kolumny: modele)
    master_df = pd.concat(drops_list, axis=1)

    # Usuwamy wiersze z NaN (upewniamy się, że klasy istnieją we wszystkich załadowanych modelach)
    master_df = master_df.dropna()
    print(f"\nUdało się złożyć macierz predykcji. Liczba wspólnych klas: {len(master_df)}")

    # Obliczamy korelacje rangową Spearmana między modelami
    corr_matrix = master_df.corr(method='spearman')

    # ==========================================
    # 4. WIZUALIZACJA (PAPER-READY)
    # ==========================================
    plt.figure(figsize=(14, 14))

    sns.set_theme(style="white")

    # Używamy clustermap, który pogrupuje podobne modele!
    g = sns.clustermap(
        corr_matrix,
        cmap='RdBu_r',  # Czerwony dla ujemnej, Niebieski dla dodatniej korelacji
        annot=True,  # Wypisuje wartości
        fmt=".2f",  # Do 2 miejsc po przecinku
        vmin=0.0, vmax=1.0,  # Zazwyczaj korelacje tu są od 0 do 1
        figsize=(16, 16),
        linewidths=.5,
        cbar_kws={"shrink": .8, "label": "Spearman Correlation"}
    )

    # Ustawienia tytułu
    g.fig.suptitle(f'Macierz Zgodności Błędów (OOD Drop) - Zbiór {TARGET_DOMAIN.upper()}',
                   y=1.02, fontsize=18, fontweight='bold')

    # Poprawa rotacji labeli, żeby były czytelne
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=12)

    # Zapisz wykres do pliku hi-res i wyświetl
    output_filename = f"correlation_matrix_{TARGET_DOMAIN}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nWykres zapisany jako: {output_filename}")
    plt.show()


if __name__ == "__main__":
    main()