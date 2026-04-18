# Dokumentacja plików konfiguracyjnych

Ten dokument opisuje strukturę i przeznaczenie plików konfiguracyjnych YAML w tym projekcie.

## 1. Definicje eksperymentów

### `experiments/experiments.yaml`

Ten plik definiuje eksperymenty do uruchomienia. Każdy eksperyment ma `name` i `type`.

- **`name`**: Unikalna nazwa eksperymentu (np. `imagenet_clean`, `imagenet_c_noise`).
- **`type`**: Typ zbioru danych do użycia.
  - `imagenet`: Czysty zbiór walidacyjny ImageNet.
  - `imagenet_c`: Zbiór danych ImageNet-C z uszkodzeniami.
- **`corruptions`**: (Tylko dla `imagenet_c`) Lista typów uszkodzeń do zastosowania (np. `gaussian_noise`, `defocus_blur`).
- **`severities`**: (Tylko dla `imagenet_c`) Lista poziomów intensywności uszkodzeń do zastosowania (od 1 do 5).

## 2. Ustawienia analizy

Te pliki konfigurują różne analizy do wykonania na wynikach eksperymentów. Znajdują się w katalogu `analysis/settings/`.

### `analysis/settings/base.yaml`

Definiuje podstawowe zadania analizy.

- **`analyses`**: Lista analiz do uruchomienia.
- **`name`**: Nazwa analizy.
- **`type`**: Typ analizy do wykonania.
  - `accuracy_drop`: Oblicza spadek dokładności między modelem bazowym a uszkodzonym zbiorem danych.
  - `fragile_class`: Identyfikuje klasy, które są najbardziej dotknięte przez uszkodzenia.
- **`content`**: Zawiera dane do analizy.
  - **`data`**: Określa pliki wejściowe.
    - `baseline`: Plik z wynikami bazowymi (np. na czystym ImageNet).
    - `degraded` lub `corruption`: Plik z wynikami dla uszkodzonego zbioru danych.
- **`output_path`**: Ścieżka do zapisania wyników analizy.

### `analysis/settings/accuracy_drop/blur.yaml`

Ten plik definiuje analizy spadku dokładności dla uszkodzeń typu "blur". Struktura jest taka sama jak w `base.yaml`, ale skoncentrowana na konkretnych modelach i uszkodzeniach.

### `analysis/settings/common_fragile.yaml`

Ten plik definiuje analizę mającą na celu znalezienie wspólnych "kruchych" klas (fragile classes) dla różnych modeli.

- **`type`**: `common_fragile_class`.
- **`content.files`**: Lista plików JSON zawierających "kruche" klasy dla każdego modelu.
- **`output_filename`**: Nazwa pliku wyjściowego JSON.

### `analysis/settings/fragile_class_overlap/settings.yaml`

Ten plik konfiguruje analizę nakładania się "kruchych" klas między różnymi modelami przy użyciu testów statystycznych.

- **`type`**:
  - `fragile_class_overlap_chi2`: Używa testu chi-kwadrat.
  - `fragile_class_overlap_fisher`: Używa dokładnego testu Fishera.
- **`content.tests`**: Lista modeli do porównania, każdy z etykietą (`label`) i ścieżką do danych (`data`).

### Pozostałe pliki w `analysis/settings/`

Pozostałe pliki YAML w podkatalogach `fragile_classes` definiują analizy "kruchych" klas dla konkretnych architektur modeli (np. `convnext_base`, `resnet50`) i typów uszkodzeń (blur, digital). Mają one tę samą strukturę co `base.yaml` z typem analizy `fragile_class`.

## 3. Ustawienia wykresów

Te pliki definiują wykresy do wygenerowania na podstawie wyników analizy. Znajdują się w katalogu `plots/`.

### `plots/plots.yaml`

Główny plik do definiowania wykresów.

- **`plots`**: Lista wykresów do wygenerowania.
- **`name`**: Nazwa wykresu.
- **`title`**: Tytuł wykresu.
- **`type`**: Typ wykresu, np.:
  - `accuracy_to_accuracy`: Wykres punktowy porównujący dokładności.
  - `accuracy_to_drop`: Wykres punktowy porównujący dokładność ze spadkiem dokładności.
  - `sorted_index`: Wykres liniowy dokładności dla różnych poziomów uszkodzeń, posortowany według dokładności bazowej.
  - `fragile_class_freq`: Wykres pokazujący częstotliwość występowania "kruchych" klas.
  - `similarity_matrix`: Macierz pokazująca podobieństwo "kruchych" klas między modelami.
  - `spearman_rank_correlation`: Mapa ciepła korelacji rang Spearmana między modelami.
- **`x_label`**, **`y_label`**: Etykiety osi.
- **`content`**: Dane do wykresu.
- **`output`**: Ścieżka do zapisania wygenerowanego wykresu.

### Pozostałe pliki w `plots/`

Większość pozostałych plików YAML w `plots/` i jego podkatalogach definiuje konkretne wykresy dla różnych modeli i typów uszkodzeń. Ich struktura jest zgodna z `plots/plots.yaml`. Służą one do organizowania dużej liczby definicji wykresów w logiczne grupy. Na przykład:
- `plots/fragile_classes/blur.yaml`: Definiuje wykresy porównujące "kruche" klasy dla uszkodzeń typu "blur".
- `plots/resnet152/imagenet_plots.yaml`: Definiuje wykresy związane z modelem ResNet-152 na zbiorze ImageNet.
- Pliki w `plots/*/acc_to_acc/` i `plots/*/acc_to_diff/` definiują wykresy punktowe porównujące dokładność z dokładnością lub dokładność ze spadkiem dokładności dla konkretnych modeli i uszkodzeń.
