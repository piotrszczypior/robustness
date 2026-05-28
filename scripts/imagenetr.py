import pandas as pd
from pathlib import Path

def load_results(path: str) -> pd.DataFrame:
    # Zakładamy, że skrypt uruchamiasz w miejscu, gdzie jest folder 'results'
    return pd.read_csv(Path("results") / path)

# 1. Wczytanie plików
imagenet = load_results("resnet152_imagenet.csv")
imagenet_a = load_results("resnet152_imagenet_a.csv")
imagenet_r = load_results("resnet152_imagenet_r.csv")

# 2. Funkcja do agregacji wyników per klasa
def get_class_accuracy(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Grupuje dane po 'synset' i oblicza średnie accuracy.
    """
    # Upewniamy się, że is_correct jest wartością numeryczną (0 lub 1)
    df['is_correct'] = df['is_correct'].astype(int)
    
    # Grupowanie po ID klasy i obliczanie średniej (accuracy)
    class_acc = df.groupby('synset')['is_correct'].mean().reset_index()
    
    # Zmiana nazwy kolumny, aby odróżnić wyniki po złączeniu tabel
    class_acc = class_acc.rename(columns={'is_correct': f'acc_{suffix}'})
    
    # Formatowanie do procentów (0-100)
    class_acc[f'acc_{suffix}'] = class_acc[f'acc_{suffix}'] * 100
    
    return class_acc

# Wyliczanie accuracy dla każdego datasetu
acc_clean = get_class_accuracy(imagenet, 'clean')
acc_ina = get_class_accuracy(imagenet_a, 'ina')
acc_inr = get_class_accuracy(imagenet_r, 'inr')

# 3. Spięcie danych (Merge)
# how='inner' zatrzyma tylko te synsety (~200), które występują w ImageNet-A i ImageNet-R
df_merged = acc_clean.merge(acc_ina, on='synset', how='inner') \
                     .merge(acc_inr, on='synset', how='inner')

# 4. Obliczenie Luki Odporności (Robustness Gap)
# Im wyższa wartość, tym bardziej model "psuje się" przy zmianie domeny na tej konkretnej klasie
df_merged['robustness_gap'] = df_merged['acc_clean'] - ((df_merged['acc_ina'] + df_merged['acc_inr']) / 2)

# 5. Wyłonienie i wyświetlenie problematycznych klas
worst_classes = df_merged.sort_values(by='robustness_gap', ascending=False)

print("TOP 15 klas z największym spadkiem skuteczności (Robustness Gap):")
print(worst_classes.head(15).round(2).to_string(index=False))

# Opcjonalny zapis ułatwiający późniejsze porównania (np. z ViTem)
# worst_classes.to_csv(Path("results") / "resnet152_robustness_analysis.csv", index=False)