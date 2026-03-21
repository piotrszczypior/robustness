# Plan Refaktoryzacji i Analizy Odporności Modeli

## Cel
Rozszerzenie obecnego systemu o automatyczną analizę porównawczą (per class/synset accuracy) między czystym ImageNet (baseline) a wariantami ImageNet-C. System ma generować wykresy typu scatter plot oraz gromadzić wyniki w ustandaryzowanej formie, zachowując przy tym obecną funkcjonalność zbierania surowych danych.

## 1. Zmiany w Konfiguracji (`experiments.yaml`)
Wprowadzenie nowej sekcji `visualizations`, która pozwoli na definiowanie porównań:

```yaml
experiments:
  - name: imagenet_clean
    type: imagenet
  - name: imagenet_c_blur
    type: imagenet_c
    corruptions: [defocus_blur]
    severities: [1, 2, 3]

visualizations:
  - type: scatter_plot
    baseline: imagenet_clean
    targets: 
      - imagenet_c_blur_defocus_blur_1
      - imagenet_c_blur_defocus_blur_2
    # Opcjonalnie: automatyczne generowanie dla wszystkich z danej grupy
    # group: imagenet_c_blur 
```

## 2. Nowy Moduł Analizy (`src/analysis.py`)
Odpowiedzialny za:
- **Agregację danych**: Wczytywanie plików CSV wygenerowanych przez `evaluate.py` i obliczanie celności per-synset.
- **Generowanie wykresów**: Tworzenie scatter plotów (Oś X: baseline accuracy, Oś Y: target accuracy).
- **Zapisywanie wyników**: Eksport zagregowanych danych do CSV (np. `results/{model}_comparison_{experiment}.csv`).

## 3. Rozszerzenie Logiki Głównej (`src/main.py`)
- Wsparcie dla listy modeli (pętla po modelach, jeśli zostanie to wskazane w konfiguracji lub argumentach).
- Po zakończeniu ewaluacji (`run_evaluation`), wywołanie modułu analizy dla każdego modelu zgodnie z sekcją `visualizations`.

## 4. Przechowywanie Danych i Logów
- **Surowe dane**: Pozostają w `results/{model}_{experiment}.csv` (minimalna agregacja).
- **Zagregowane dane**: `results/analysis/{model}_per_class.csv`.
- **Wykresy**: `results/plots/{model}_{baseline}_vs_{target}.png`.
- **Logi**: Rozszerzenie `logs/run.log` o informacje o postępie generowania wykresów.

## 5. Workflow (Kroki Implementacji)
1. **Zaktualizowanie `src/experiment.py`**: Dodanie parsera dla sekcji `visualizations`.
2. **Implementacja `src/analysis.py`**:
    - Funkcja `get_per_class_accuracy(df)` -> zwraca Series/DataFrame z synset jako indeksem.
    - Funkcja `plot_scatter(baseline_df, target_df, title, save_path)`.
3. **Modyfikacja `src/main.py`**: Integracja pętli ewaluacja -> analiza.
4. **Testy**: Uruchomienie dla ResNet152 na małej próbce danych w celu weryfikacji wykresów.

## Zachowanie Funkcjonalności
- Skrypty `extract_imagenet.sh` i `setup.sh` pozostają bez zmian.
- Format wyjściowy surowych danych CSV z `evaluate.py` pozostaje identyczny, co zapewnia kompatybilność wsteczną z obecnymi notebookami.
