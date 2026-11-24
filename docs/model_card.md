# Model Card

> Scaffold for Sprint 3 — do not implement before this sprint.
> Minimalną Model Card wypełnia zespół w Sprincie 3, a aktualizuje w Sprincie 6.

## Model

- Typ modelu i jego wersja (alias `production` w W&B): AutoGluon `TabularPredictor`,
  eksperyment `ag_gbm` (preset `medium_quality`, tylko LightGBM, limit 120 s, najlepszy model
  `WeightedEnsemble_L2`). W W&B: `iranian-churn-model:v0` z aliasem `production`
  (projekt `ASI-s31686-s30754-s30383`).
- Data utworzenia i autor runu: 24.11.2026. Trening i run `ag_gbm`: Michał Sebastianowicz.
  Zapis artefaktu `production` (run `publish-ag_gbm`): Bartosz Wołosz.
- Jak sprawdzić: `python scripts/smoke_model.py --wandb` pobiera `production`, robi predykcję
  dla 5 wierszy i sprawdza schemat odpowiedzi.

## Przeznaczenie

- Problem (binary classification) i klasa pozytywna: czy klient operatora komórkowego odejdzie
  (`Churn`). Klasa pozytywna `1` = klient odszedł.
- Użytkownik systemu i decyzja, którą wspiera model: dział obsługi klienta. Model wskazuje
  klientów, którzy mogą odejść, żeby zdążyć się z nimi skontaktować. Predykcja jest robiona
  na koniec 9. miesiąca z danych z miesięcy 1–9, a wynik dotyczy końca 12. miesiąca.
- Zastosowania, do których model nie jest przeznaczony: automatyczne decyzje bez człowieka
  (np. zmiana ceny albo blokada usługi), klienci innych operatorów i z innych lat.

## Dane

- Dataset (id z `docs/datasets.md`) i fingerprint danych: `iranian_churn` (UCI, CC BY 4.0),
  fingerprint `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1`.
- Strategia splitu z kontraktu i liczebności zbiorów: `stratified_random`, 80/20,
  `random_state` 42. Po usunięciu 300 duplikatów zostaje 2850 rekordów: 2280 trening
  i 570 test (w teście 89 klientów, którzy odeszli). Model używa 11 cech z `project.yaml`.

## Metryki

- Metryka główna na zbiorze testowym, w porównaniu z DummyClassifier: F1 0,86
  (DummyClassifier 0,00, baseline LogisticRegression 0,60).
- Pozostałe metryki i macierz pomyłek: precision 0,87, recall 0,85, accuracy 0,96.
  TN 470, FP 11, FN 13, TP 76.
- Parametry operacyjne (pomiary z W&B): predykcja 100 wierszy 10 ms, pakiet 2,4 MB,
  pamięć około 91 MB po wczytaniu i pierwszej predykcji.

## Ograniczenia i ryzyka

- Ograniczenia danych (z `docs/datasets.md`): mały zbiór (3150 rekordów, w teście tylko
  89 klientów z klasą 1), więc kilku klientów zmienia F1 o kilka punktów. Dane pochodzą
  od jednego operatora z Iranu. Część cech to kategorie zakodowane liczbami.
- Znane błędy i sytuacje, w których model zawodzi: na teście nie wykrył 13 z 89 klientów,
  którzy odeszli, i dał 11 fałszywych alarmów. Nie był sprawdzany na danych spoza tego zbioru.
  Do wczytania potrzebny jest AutoGluon, a pakiet zawiera kopię danych treningowych
  zapisaną przez AutoGluona (`model/utils/data/`).

## Wersje

- Wersje bibliotek modelu i git SHA treningu: autogluon.tabular 1.6.3, scikit-learn 1.7.2,
  pandas 2.3.3, Python 3.11.18. Git SHA treningu `846acdeef46bef45f4151ae22957d5ba26889a41`
  (branch `s3/wandb-experiments`). Pełna lista jest w `metadata.json` pakietu.
