# ADR-1: Granice przetwarzania danych

- Status: zaakceptowana
- Data: 2026-11-05

## Decyzja

Rozdzielamy przetwarzanie danych na trzy warstwy:

1. **Input normalization** pozostaje wspólne dla treningu i przyszłego API.
   Obejmuje wyłącznie operacje bezpieczne dla pojedynczego nowego rekordu, takie jak
   ujednolicanie nazw i białych znaków.
2. **Training-only filtering** działa wyłącznie w pipeline przygotowania danych
   treningowych. Usuwa rekordy bez targetu i duplikaty, ale nie jest wykonywane podczas
   predykcji.
3. **Model preprocessing** jest częścią sklearn `Pipeline`. Imputacja, skalowanie
   i kodowanie są dopasowywane na danych treningowych i zapisywane razem z modelem.

## Uzasadnienie

Podział zapobiega wyciekowi informacji i gwarantuje, że API używa tych samych reguł
normalizacji co trening. Umieszczenie operacji wymagających `fit()` w modelu zapewnia
identyczne przetwarzanie podczas treningu i predykcji.
