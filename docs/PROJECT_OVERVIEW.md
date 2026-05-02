
# Przegląd Projektu — Analiza Odporności Modeli Wizyjnych

Ten dokument przedstawia ogólny przegląd projektu, jego cele, architekturę oraz kluczowe możliwości.

## 1. Cel Projektu

Głównym celem tego frameworka jest **systematyczna analiza odporności (robustness) modeli głębokiego uczenia** przeznaczonych do klasyfikacji obrazów. Projekt umożliwia ocenę, jak różne architektury modeli (np. `ResNet`, `ConvNeXT`, `Vision Transformer`) radzą sobie z typowymi uszkodzeniami obrazu, takimi jak szum, rozmycie, mgła czy artefakty cyfrowe.

Badania prowadzone są głównie z wykorzystaniem benchmarku **ImageNet-C**, który dostarcza standardowy zestaw 15 typów uszkodzeń w 5 poziomach intensywności.

## 2. Możliwości Frameworka

Framework został zbudowany w sposób modułowy i jest sterowany głównie przez pliki konfiguracyjne YAML. Główne komponenty funkcjonalne to:

- **`evaluate` (Ocena)**: Uruchamia ewaluację określonych modeli na wybranych zestawach danych (czystych lub uszkodzonych). Konfiguracja eksperymentów znajduje się w `experiments/experiments.yaml`. Wyniki zapisywane są w formacie CSV.

- **`analyze` (Analiza)**: Przetwarza surowe wyniki w celu głębszej analizy. Potrafi m.in.:
  - Obliczać spadek dokładności (`accuracy_drop`) w stosunku do wyników na czystym zbiorze danych.
  - Identyfikować **"kruche klasy" (`fragile_classes`)**, czyli klasy, których rozpoznawanie jest najbardziej podatne na degradację przez dany typ uszkodzenia.
  - Porównywać zbiory "kruchych klas" między różnymi modelami w celu znalezienia podobieństw i różnic w ich słabościach.

- **`plot` (Wizualizacja)**: Generuje szeroki wachlarz wizualizacji na podstawie przetworzonych danych, co ułatwia interpretację wyników. Przykładowe wykresy to:
  - Wykresy punktowe porównujące dokładność modeli.
  - Macierze podobieństwa dla "kruchych klas".
  - Mapy ciepła korelacji (np. korelacji rang Spearmana) między zachowaniami modeli.
  - Wykresy słupkowe i liniowe pokazujące zmiany dokładności.

- **`xai` (Wyjaśnialna AI)**: Moduł przeznaczony do uruchamiania technik XAI (Explainable AI), prawdopodobnie w celu wizualizacji, na czym skupiają się modele podczas klasyfikacji obrazów czystych i uszkodzonych (np. przy użyciu Grad-CAM).

- **`sankey` (Diagramy Sankeya)**: Tworzy diagramy Sankeya, które mogą wizualizować przepływ błędnych klasyfikacji — jak próbki z jednej klasy są mylone z innymi po wprowadzeniu uszkodzeń.

## 3. Struktura Projektu i Użycie

Framework jest obsługiwany z linii poleceń za pomocą centralnego skryptu `src/main.py`, który parsuje argumenty i uruchamia odpowiednie zadania.

- **Główny skrypt**: `run.sh` jest prostym wrapperem, który wywołuje `src/main.py` z odpowiednim interpreterem Python.
- **Konfiguracja**:
  - `experiments/experiments.yaml`: Definiuje, jakie eksperymenty (modele, uszkodzenia, poziomy intensywności) mają zostać przeprowadzone.
  - `plots/plots.yaml` oraz inne pliki w tym katalogu: Definiują, jakie wykresy mają zostać wygenerowane.
  - `analysis/settings/*.yaml`: Definiują zadania analityczne.
- **Dane wejściowe**: Zbiory danych (np. ImageNet) powinny znajdować się w katalogu `data/`.
- **Wyniki**:
  - `results/`: Surowe wyniki ewaluacji w plikach CSV.
  - `analysis/`: Wyniki analiz, np. listy "kruchych klas" w plikach JSON.
  - `images/` i `plots/`: Wygenerowane wykresy i wizualizacje.

## 4. Aktualne Badania

Na podstawie struktury plików i konfiguracji można wywnioskować, że obecne badania koncentrują się na:

- **Porównaniu architektur**: Analizowane są modele takie jak `resnet50`, `resnet152`, `efficientnet_b4`, `convnext_base` i `vit_b_16`.
- **Analizie wpływu różnych typów uszkodzeń**: Eksperymenty są pogrupowane na kategorie: `blur` (rozmycia), `noise` (szumy), `digital` (cyfrowe) itd.
- **Identyfikacji słabości modeli**: Głęboka analiza "kruchych klas" (fragile classes) oraz metryki `accuracy drop`. W ramach projektu, "kruche klasy" są definiowane jako te, które na czystym zbiorze danych osiągnęły dokładność (accuracy) na poziomie co najmniej 80%, a pod wpływem korupcji ich dokładność spadła poniżej 50%. Badanie podobieństw w tych klasach między różnymi modelami ma na celu odkrycie fundamentalnych słabości w sposobie, w jaki obecne modele wizyjne generalizują wiedzę.
- **Korelacji w zachowaniach modeli**: Analiza korelacji rang Spearmana wskazuje na badanie, czy różne modele popełniają błędy w podobny sposób, co mogłoby świadczyć o wspólnych, fundamentalnych problemach.
