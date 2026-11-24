# ADR-2: Wybór modelu produkcyjnego i jego postać jako artefaktu

- **Status:** proponowany
- **Sprint:** S3
- **Autor:** Bartosz Wołosz
- **Recenzent:** Michał Sebastianowicz

## Kontekst

W Sprincie 4 API ma używać jednego gotowego modelu, bez ponownego treningu. Musimy wybrać,
który model idzie dalej, i ustalić, w jakiej postaci go przekazujemy.

Porównaliśmy baseline ze Sprintu 2 i dwie konfiguracje AutoGluona. Wszystkie runy w W&B mają
te same dane (ten sam fingerprint), ten sam kod (git `846acde`) i ten sam zbiór testowy
(570 rekordów, w tym 89 klientów, którzy odeszli).

Kryteria wyboru: F1 na teście (metryka główna), czas predykcji, rozmiar pakietu i pamięć
po wczytaniu.

## Rozważone opcje

1. **baseline_logreg**: LogisticRegression ze Sprintu 2.
2. **ag_gbm**: AutoGluon, preset `medium_quality`, tylko LightGBM, limit 120 s.
3. **ag_trees_bagged**: AutoGluon, preset `good_quality`, LightGBM, CatBoost, Random Forest
   i Extra Trees z baggingiem, limit 180 s.

## Decyzja

Wybieramy **ag_gbm**.

| Model | F1 | Recall | Predykcja 100 wierszy | Rozmiar pakietu | Pamięć po wczytaniu i 1. predykcji |
|---|---|---|---|---|---|
| baseline_logreg | 0,60 | 0,47 | 1,8 ms | 0,01 MB | 58 MB |
| **ag_gbm** | **0,86** | **0,85** | **10,0 ms** | **2,4 MB** | **91 MB** |
| ag_trees_bagged | 0,89 | 0,89 | 3,8 ms | 35,3 MB | 90 MB |

Skąd liczby: runy W&B w projekcie `ASI-s31686-s30754-s30383` (pipeline `experiments`).

Dlaczego: ag_gbm ma prawie takie samo F1 jak najlepszy model (różnica 0,02, czyli 3 klientów
na 89), a jego pakiet jest 15 razy mniejszy. Czas predykcji obu modeli AutoGluona to mniej
niż 0,1 ms na klienta, a pamięć jest taka sama, więc te kryteria niczego nie rozstrzygają.

Postać artefaktu: katalog z `save_model_package` (`model/` + `metadata.json`) zapisany w W&B
jako `iranian-churn-model` (typ `model`) z aliasem `production`, teraz wersja `v0`.
API i smoke test pobierają model po aliasie, a nie po numerze runu.

## Odrzucone alternatywy

- **baseline_logreg**: najmniejszy i najszybszy, ale wykrywa tylko 42 z 89 klientów,
  którzy odeszli (recall 0,47).
- **ag_trees_bagged**: najlepsze F1, ale przewaga to tylko 3 klientów, co na tak małym teście
  może być przypadkiem. Pakiet ma 35 MB, bo trzyma też modele CatBoost, RF i XT.
- **Sam plik modelu albo wskazywanie numeru runu**: bez `metadata.json` API nie wie, jakich
  cech oczekuje model i skąd on pochodzi. Numer runu trzeba by zmieniać w API przy każdej
  zmianie modelu, a alias `production` po prostu przenosimy na nową wersję.

## Konsekwencje

- Ułatwia: API w Sprincie 4 pobiera `iranian-churn-model:production` i wczytuje go przez
  `load_model_package`, bez wiedzy o treningu. Ta ścieżka działa już w
  `scripts/smoke_model.py --wandb`.
- Utrudnia: do wczytania modelu potrzebny jest AutoGluon 1.6.3, więc obraz Dockera
  w Sprincie 4 będzie większy niż z samym scikit-learn.
- Do zrobienia później: nowy model zapisujemy przez `scripts/publish_model.py` jako nową
  wersję artefaktu z aliasem `production`.

## Historia

| Data | Zmiana |
|---|---|
| 2026-11-24 | utworzenie |
