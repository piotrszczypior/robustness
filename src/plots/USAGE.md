# Dokumentacja CLI Wykresów

Wszystkie komendy są dostępne jako pod-komendy głównego narzędzia `plot`. Poniżej znajduje się opis każdego typu wykresu oraz instrukcja jego użycia.

## Spis treści
- [Accuracy Scatter](#accuracy-scatter)
- [Class Degradation](#class-degradation)
- [Fragile Classes Comparison](#fragile-classes-comparison)
- [Fragile Similarity](#fragile-similarity)
- [Spearman Rank](#spearman-rank)
- [Spearman (Domain)](#spearman-domain)
- [Jaccard (Domain)](#jaccard-domain)
- [Embeddings](#embeddings)
- [Violin Plot](#violin-plot)
- [Violin RmCE](#violin-rmce)
- [Barcode RmCE](#barcode-rmce)

---

## Accuracy Scatter
`accuracy_scatter` - Generuje wykres rozrzutu (scatter plot) porównujący dokładność (accuracy) między dwoma zestawami wyników.

**Użycie:**
```bash
python main.py plot accuracy_scatter x_file y_file [--mode {default,drop}] [--data DATA]
```

**Argumenty:**
- `x_file`: Plik CSV dla osi X (np. wyniki bazowe).
- `y_file`: Plik CSV dla osi Y (np. wyniki modelu po degradacji).
- `--mode`: Tryb wykresu. `default` porównuje bezpośrednio dwie dokładności, `drop` pokazuje spadek dokładności (Y - X).
- `--data`: Katalog z danymi (domyślnie: `results/`).

---

## Class Degradation
`class_degradation` - Wyświetla degradację dokładności klas, posortowanych według ich wyników w modelu bazowym.

**Użycie:**
```bash
python main.py plot class_degradation --baseline-label LABEL --baseline-data DATA --degraded-label LABEL --degraded-data DATA [--data DATA]
```

**Argumenty:**
- `--baseline-label`: Etykieta dla serii bazowej (np. "Clean").
- `--baseline-data`: Plik CSV z wynikami bazowymi.
- `--degraded-label`: Etykieta dla serii zdegradowanej (np. "Blurred").
- `--degraded-data`: Plik CSV z wynikami po degradacji.
- `--data`: Katalog z danymi.

---

## Fragile Classes Comparison
`fragile_classes_comparison` - Porównuje zestawy klas "kruchych" (fragile) między różnymi modelami lub degradacjami.

**Użycie:**
```bash
python main.py plot fragile_classes_comparison --files FILE [FILE ...] --names NAME [NAME ...] [--mode {default,freq}] [--data DATA]
```

**Argumenty:**
- `--files`: Lista plików JSON zawierających analizę klas kruchych.
- `--names`: Nazwy odpowiadające podanym plikom (do wyświetlenia na wykresie).
- `--mode`: `default` (binarna mapa kruchych klas), `freq` (częstotliwość występowania kruchych klas).

---

## Fragile Similarity
`fragile_similarity` - Generuje macierz podobieństwa (Jaccard Index) zestawów kruchych klas.

**Użycie:**
```bash
python main.py plot fragile_similarity --files FILE [FILE ...] --names NAME [NAME ...] [--data DATA]
```

**Argumenty:**
- `--files`: Lista plików JSON z analizą klas.
- `--names`: Etykiety dla osi macierzy.

---

## Spearman Rank
`spearman_rank` - Wyświetla macierz korelacji rang Spearmana dla spadków dokładności między modelami.

**Użycie:**
```bash
python main.py plot spearman_rank --files FILE [FILE ...] --names NAME [NAME ...] [--data DATA]
```

**Argumenty:**
- `--files`: Lista plików JSON z analizą klas.
- `--names`: Etykiety dla macierzy korelacji.

---

## Spearman (Domain)
`spearman` - Zaawansowane wykresy korelacji Spearmana specyficzne dla ImageNet-C i grup korupcji.

**Użycie:**
```bash
python main.py plot spearman [--metric {drop,rank}] [--mode {default,average}] [--data DATA] [--corruptions CORR ...] [--severities SEV ...]
```

**Argumenty:**
- `--metric`: Metryka korelacji (`drop` dla spadku dokładności, `rank` dla rang dokładności).
- `--mode`: `default` (poszczególne korupcje), `average` (uśrednione wyniki dla grup korupcji).
- `--corruptions`: Lista konkretnych korupcji do analizy.
- `--severities`: Lista poziomów intensywności (1-5).

---

## Jaccard (Domain)
`jaccard` - Analiza pokrycia (Jaccard overlap) najsłabszych klas w domenie ImageNet-C.

**Użycie:**
```bash
python main.py plot jaccard [--top-k K] [--data DATA] [--tail {worst,best}] [--corruptions CORR ...] [--severities SEV ...]
```

**Argumenty:**
- `--top-k`: Liczba klas branych pod uwagę (domyślnie 50).
- `--tail`: Czy analizować najsłabsze (`worst`) czy najlepsze (`best`) klasy.

---

## Embeddings
`embeddings` - Generuje projekcje embeddingów (np. t-SNE, UMAP).

**Użycie:**
```bash
python main.py plot embeddings [--settings SETTINGS] [--data DATA] [--projection PROJECTION]
```

**Argumenty:**
- `--settings`: Ścieżka do pliku YAML z konfiguracją projekcji (domyślnie: `src/plots/embeddings/settings.yaml`).
- `--projection`: Opcjonalne wymuszenie konkretnego typu projekcji.

---

## Violin Plot
`violin` - Wykresy skrzypcowe rozkładu dokładności klas.

**Użycie:**
```bash
python main.py plot violin [--mode {single,collage}] [--data DATA] [--models MODELS ...] [--corruptions CORR ...] [--severities SEV ...]
```

**Argumenty:**
- `--mode`: `single` (osobny wykres dla każdego modelu), `collage` (wszystkie modele na jednym wykresie).

---

## Violin RmCE
`violin_rmce` - Wykresy skrzypcowe rozkładu Relative mean Class Error (RmCE).

**Użycie:**
```bash
python main.py plot violin_rmce [--data DATA] [--models MODELS ...] [--corruptions CORR ...] [--severities SEV ...]
```

---

## Barcode RmCE
`barcode_rmce` - Wykresy kodów kreskowych (barcode) identyfikujące klasy kruchre na podstawie RmCE > 1.0.

**Użycie:**
```bash
python main.py plot barcode_rmce [--data DATA] [--models MODELS ...] [--corruptions CORR ...] [--severities SEV ...]
```
