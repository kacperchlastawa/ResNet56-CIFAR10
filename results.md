# Wyniki: ResNet-56 na CIFAR-10 (Colab, GPU, AMP fp16)

Ustawienia zgodne z artykułem (He et al., 1512.03385, sekcja 4.2): n=9 (56 warstw, 0.853M parametrów), SGD, momentum 0.9,
weight decay 1e-4, batch 128, lr 0.1 → 0.01 @32k → 0.001 @48k, koniec @64k iteracji, inicjalizacja He, BN, bez dropoutu,
augmentacja (pad 4 + crop 32×32 + flip), odjęcie średniej per-piksel, trening na 45k, walidacja na 5k.
Czas treningu: 2786 s (~46 min), ok. 0.043 s/iteracja.

## Wynik końcowy (iteracja 64 000)

| Zbiór | Błąd [%] |
|---|---|
| Train (średnia z ostatnich 500 iteracji, z augmentacją) | 0.08 |
| Walidacja (5k) | 7.08 |
| **Test (10k)** | **7.52** |
| Artykuł, ResNet-56 (Tab. 6) | 6.97 |

Różnica względem artykułu: +0.55 p.p.

## Przebieg treningu (wybrane punkty)

| Iteracja | lr | train_err [%] | val_err [%] |
|---|---|---|---|
| 500 | 0.1 | 76.65 | 63.74 |
| 5 000 | 0.1 | 16.01 | 22.40 |
| 10 000 | 0.1 | 10.30 | 15.04 |
| 20 000 | 0.1 | 7.66 | 13.62 |
| 31 500 | 0.1 | 6.14 | 14.02 |
| 32 000 | 0.01 | 5.96 | 12.80 |
| 32 500 | 0.01 | 2.69 | 7.62 |
| 40 000 | 0.01 | 0.39 | 7.12 |
| 48 000 | 0.001 | 0.20 | 7.16 |
| 56 000 | 0.001 | 0.08 | 6.96 |
| 61 000 | 0.001 | 0.07 | 6.84 (minimum) |
| 64 000 | 0.001 | 0.08 | 7.08 |

## Obserwacje
- Spadek lr w 32k daje skok błędu val z ~12–14% do ~7.6%. Po 33k val_err stoi w miejscu (6.84–7.5%), a train_err spada do ~0.1%.
- Przy lr = 0.1 val_err mocno oscyluje (10.6–19%). To normalne dla dużego lr i BN.
- Po 33k iteracjach dalszy trening praktycznie nie poprawia val. Początkowy skok błędu to efekt ścieżki, a nie sygnał przeuczenia na końcu.
- Ostrzeżenie `lr_scheduler.step() before optimizer.step()` jest nieszkodliwe: `GradScaler` (AMP) pomija pierwszy krok optymalizatora przy inf/NaN.

---

# Porównanie z siecią plain-56 (bez połączeń rezydualnych)

Identyczne ustawienia i ziarno, zmienia się tylko `residual=False` (27 shortcutów wyłączonych, takie same głębokość, szerokość i liczba parametrów).
Czas: 2673 s (ResNet-56: 2783 s).

| Model | train_err (ost. 500 it.) [%] | val_err [%] | **test_err [%]** |
|---|---|---|---|
| ResNet-56 | 0.08 | 7.08 | **7.52** |
| Plain-56 | 4.85 | 14.46 | **15.50** |

Wybrane punkty (train_err / val_err [%]):

| Iteracja | lr | ResNet-56 | Plain-56 |
|---|---|---|---|
| 500 | 0.1 | 76.65 / 63.74 | 89.35 / 88.02 |
| 5 000 | 0.1 | 16.01 / 22.40 | 53.51 / 53.94 |
| 20 000 | 0.1 | 7.66 / 13.62 | 22.89 / 30.10 |
| 32 500 | 0.01 | 2.69 / 7.62 | 13.88 / 15.70 |
| 48 000 | 0.001 | 0.20 / 7.16 | 7.34 / 15.10 |
| 64 000 | 0.001 | 0.08 / 7.08 | 4.85 / 14.46 |

## Wnioski
- Plain-56 ma **wyższy błąd treningowy** (4.85% wobec 0.08%), więc nie chodzi o przeuczenie, tylko o trudniejszą optymalizację. Dokładnie to opisuje problem degradacji z artykułu.
- Plain-56 uczy się wolniej od początku (po 500 iteracjach train_err 89% wobec 77%) i ma gorszy błąd testowy o ok. 8 p.p. (15.50% wobec 7.52%).
- Połączenia rezydualne poprawiają wynik przy tej samej głębokości i liczbie parametrów.
- Ograniczenie: pojedyncze uruchomienie, bez plain-20. Pełne potwierdzenie degradacji z artykułu (plain-56 gorszy niż płytsza plain-20) wymagałoby jeszcze plain-20.
