# ADR-1: Gdzie żyje przetwarzanie danych

- **Status:** proponowany
- **Sprint:** S2
- **Autor:** Bartosz Wołosz
- **Recenzent:** Michał Sebastianowicz

## Kontekst

Dane Iranian Churn przechodzą przez trzy rodzaje przetwarzania: input normalization,
training-only filtering i model preprocessing. W Sprincie 1 robiliśmy to w notebookach,
a od Sprintu 2 trening idzie przez `kedro run`. W Sprincie 4 dojdzie API, które dostaje
jednego klienta naraz i nie ma dostępu do całego zbioru ani do targetu.

Musimy więc ustalić, gdzie każde przetwarzanie ma się wykonywać, tak żeby trening
i późniejsze API przetwarzały dane tak samo i żeby nic z danych testowych nie przeciekło
do treningu.

## Rozważone opcje

1. **Trzy osobne miejsca** — input normalization w `normalize_input()` (`data.py`),
   training-only filtering w `filter_training_rows()` (`data.py`), model preprocessing
   w sklearn `Pipeline` (`model.py`). Kedro tylko łączy te funkcje w nody.
2. **Całe przygotowanie danych w pipeline'ie `data_processing`** — imputacja, skalowanie
   i kodowanie jako osobne nody Kedro przed treningiem.
3. **Wszystko, co się da, w `normalize_input()`** — razem z usuwaniem duplikatów
   i uzupełnianiem braków.

## Decyzja

Wybieramy opcję 1.

- **Input normalization** (`normalize_input()`, w pipeline'ie node `normalize_raw_data_node`
  razem z ujednoliceniem nazw kolumn): działa na pojedynczym rekordzie, nie używa targetu
  ani statystyk zbioru. Tę samą funkcję wywoła API. W naszych danych nie ma tekstu
  ani braków, więc zostają domyślne reguły. Zer w kolumnach z korzystaniem z usług nie
  zamieniamy na braki, bo oznaczają nieaktywnych klientów, a nie brak danych.
- **Training-only filtering** (`filter_training_rows()`, node `filter_training_rows_node`):
  usuwa rekordy bez targetu i dokładne duplikaty. Działa tylko przy przygotowaniu danych
  do treningu, nigdy w API, bo przy predykcji przychodzi jeden klient i nie ma czego
  deduplikować.
- **Model preprocessing** (imputacja medianą, skalowanie, one-hot) zostaje w sklearn
  `Pipeline` z `build_baseline_pipeline()`. Dopasowuje się tylko na `X_train` i zapisuje
  razem z modelem w `data/06_models/model.pkl`.

Kolejność nodów w `data_processing` to: input normalization → walidacja kontraktu →
training-only filtering. Normalizacja idzie przed walidacją, bo plik źródłowy ma podwójne
spacje w nazwach kolumn i bez ich ujednolicenia walidacja nie znalazłaby wymaganych kolumn.

| Wielkość | Wartość | Skąd pochodzi |
|---|---|---|
| Dokładne duplikaty usunięte przez training-only filtering | 300 (zostaje 2850 z 3150 rekordów) | `kedro run --pipeline data_processing`, notebook 01 (punkt 7) |
| Klienci z zerami w `Seconds of Use`, `Frequency of use`, `Distinct Called Numbers` | 154 (zostają jako 0) | notebook 01 (punkt 6) |
| F1 na zbiorze testowym: LogisticRegression vs DummyClassifier | 0,60 vs 0,00 (n_test = 570) | `metrics.json` ze Sprintu 1 |

## Odrzucone alternatywy

- **Opcja 2 (preprocessing jako nody Kedro):** odrzucona, bo dopasowany imputer, skaler
  i encoder nie byłyby częścią zapisanego modelu. API musiałoby odtwarzać te kroki osobno
  i łatwo o różnicę między treningiem a predykcją. Do tego łatwo przez pomyłkę dopasować
  je na całych danych przed splitem, co daje wyciek z testu.
- **Opcja 3 (wszystko w `normalize_input()`):** odrzucona, bo uzupełnianie braków medianą
  wymaga statystyk całego zbioru, a usuwanie duplikatów wymaga patrzenia na inne rekordy.
  Żadna z tych operacji nie działa dla jednego klienta w API. Testy z `tests/test_project.py`
  sprawdzają, że `normalize_input()` daje ten sam wynik dla rekordu osobno i w całym zbiorze.

## Konsekwencje

- Ułatwia: API w Sprincie 4 wywoła tylko `normalize_input()` i wczyta `model.pkl`, który
  ma w sobie cały model preprocessing. Logika jest w jednym miejscu (`data.py`, `model.py`),
  a notebooki, Kedro i API z niej korzystają zamiast ją kopiować.
- Utrudnia: zmiana reguł normalizacji wymaga ponownego treningu, bo model był uczony
  na danych po starej normalizacji.
- Do zrobienia później: w Sprincie 4 API musi wywoływać `normalize_input()` przed modelem,
  a nie własną kopię tych reguł.

## Historia

| Data | Zmiana |
|---|---|
| 2026-11-05 | utworzenie na podstawie szkicu z PR #6 (`docs/adr/adr-1-processing-boundaries.md`), uzupełnienie według szablonu |
