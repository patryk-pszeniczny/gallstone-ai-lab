DOCS_TEXT = r'''\
# GallStone MLP — Dokumentacja techniczna (pełna)

> **Wersja:** Stan na 2025-10-24. Aplikacja: MLP (numpy only) + Tkinter/ttk GUI + matplotlib, bez scikit-learn.


Ten dokument opisuje **każdy moduł, klasę i funkcję** w aplikacji, wraz z podstawami
teoretycznymi użytych algorytmów. Zawiera także wskazówki dot. stosowania, złożoność
obliczeniową, dobre praktyki i najczęstsze pułapki.

## Spis treści

- 1. Architektura i moduły
- 2. Szybki start
- 3. Teoria i pojęcia (neurony, aktywacje, strata, SGD, ES, StepLR, CV)
- 4. API modułów `core/*` (dataset_io, preprocess, metrics, platt, mlp, trainer, analysis, plots)
- 5. API GUI `gui/*` (BaseTab, DataTab, TrainTab, PredictTab, PlotsTab, AnalysisTab, ModelTab, DocsTab)
- 6. Przepływ pracy i dobre praktyki
- 7. Testowanie, walidacja, debugowanie
- 8. Częste problemy i rozwiązania
- 9. Licencja i podziękowania

## 1. Architektura i moduły


Aplikacja jest podzielona na **rdzeń (`core/`)** oraz **interfejs graficzny (`gui/`)**.
Struktura:

```
gallstone_app/
├─ main.py
├─ assets/
│  └─ docs_text.py          # (ten plik) – rozbudowana dokumentacja w Markdown
├─ core/
│  ├─ state.py              # pojedyncze źródło stanu aplikacji
│  ├─ dataset_io.py         # wczytywanie CSV/XLSX (z fallback parserem XML)
│  ├─ preprocess.py         # split stratyfikowany, z-score
│  ├─ metrics.py            # metryki, krzywe ROC/PR, reliability, itd.
│  ├─ platt.py              # kalibracja Platta (fit/apply)
│  ├─ mlp.py                # MLP od zera (numpy), 1-2 warstwy ukryte
│  ├─ trainer.py            # pętla treningowa, early stopping, step LR
│  ├─ analysis.py           # tabela FP/FN, permutation importance
│  └─ plots.py              # generatory wykresów (matplotlib Figure)
└─ gui/
   ├─ base.py               # wspólne narzędzia GUI
   ├─ data_tab.py           # karta „Dane”
   ├─ train_tab.py          # karta „Trening”
   ├─ predict_tab.py        # karta „Predykcja”
   ├─ plots_tab.py          # karta „Wykresy”
   ├─ analysis_tab.py       # karta „Analiza”
   ├─ model_tab.py          # karta „Model”
   └─ docs_tab.py           # karta „Dokumentacja” (render tej treści)
```
## 2. Szybki start


1. Uruchom `python main.py`.
2. W zakładce **Dane** wczytaj plik CSV/XLSX z kolumną celu `Gallstone Status` (0/1).
3. W **Treningu** wybierz hiperparametry → **Podziel + Z-score** → **START**.
4. Obejrzyj logi i wykresy (**Wykresy**: Loss/ROC/PR/CM). Dostosuj próg w CM.
5. W **Analizie** sprawdź FP/FN, *Permutation Importance*, *Reliability diagram*.
6. W **Modelu** zapisz/wczytaj model, ucz kalibrację Platta, wykonaj batch-predykcję.
## 3. Teoria i pojęcia

### Neuron (perceptron ciągły)


**Neuron** w warstwie ukrytej/wyjściowej oblicza:  
\\( z = x^\top w + b \\) oraz **aktywację** \\( a = \phi(z) \\).  
Dla warstwy wyjściowej klasyfikacji binarnej stosujemy **sigmoidę**:  
\\( \sigma(z) = \frac{1}{1+e^{-z}} \\) → wynik interpretujemy jako \\( P(y=1\mid x) \\).

**ReLU** (Rectified Linear Unit): \\( \text{ReLU}(z) = \max(0,z) \\) – szybka, niweluje problem zanikania gradientu dla dodatnich aktywacji.  
**Tanh**: \\( \tanh(z) = \frac{e^z-e^{-z}}{e^z+e^{-z}} \\), wartości w \\([-1,1]\\).

**Wagi** inicjalizujemy metodami *He* (dla ReLU) lub *Xavier/Glorot* (dla tanh/sigmoid), by utrzymać wariancję sygnału w głąb sieci.

### Binary Cross-Entropy (BCE)


Strata używana do uczenia klasyfikatora binarnego:
\\[
  \mathcal{L}_\\text{BCE}(y, \hat{p}) = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right].
\\]
Zapewnia silny gradient dla przewidywań bliskich 0/1 i dobrze współgra z wyjściową sigmoidalną.

### SGD (mini-batch), Backprop i L2


**Backpropagacja** wyprowadza gradienty po wagach poprzez regułę łańcuchową.  
**Mini-batch SGD**: dzielimy dane na paczki, aktualizujemy parametry po każdej paczce.  
**L2 (weight decay)** dodaje do gradientu składnik \\( \lambda W \\), zapobiega nadmiernym wartościom wag.

### Early stopping


Zatrzymuje trening, gdy metryka walidacyjna (np. `val_loss`) nie poprawia się przez `patience` epok. Po zakończeniu przywracane są **najlepsze wagi** (najniższy `val_loss`).

### Harmonogram uczenia (Step LR decay)


Co `step_every` epok modyfikujemy LR: \\( \text{lr} \leftarrow \gamma\cdot\text{lr} \\) (np. \\( \gamma=0.5 \\)). W praktyce stabilizuje trening i pozwala schodzić do minimum.

### Stratyfikowany podział i K-fold CV


**Stratyfikacja** zachowuje proporcje klas w train/test.  
**K-fold CV (stratyfikowany)** dzieli dane na K częsci, trenuje K modeli (zostawiając jeden fold jako walidację), uśrednia metryki. Pomaga ocenić stabilność i wariancję wyniku.

### ROC/AUC oraz PR/AUPRC


**ROC**: krzywa TPR vs FPR przy zmianie progu. **AUC** to pole pod ROC (integracja trapezami).  
**PR**: Precision vs Recall. **AUPRC** często lepsze przy klasie rzadkiej.  
W implementacji: generujemy wektor progów \\([0,1]\\), wyliczamy punkty, sortujemy i całkujemy numerycznie.

### Kalibracja prawdopodobieństw i Platt scaling


**Reliability diagram** porównuje przewidziane prawdopodobieństwa do częstości empirycznych w kubełkach.  
**Platt scaling** uczy funkcję \\( \sigma(a s + b) \\) mapującą skory \\( s \\) na lepiej skalibrowane prawdopodobieństwa. Tu realizacja bez scikit-learn (optymalizacja 2D Newtonem).

## 4. API rdzenia (`core/*`)

### 4.1. `core/dataset_io.py`


**Funkcje:**

- `read_dataset(path: str) -> pd.DataFrame`  
  Wczytuje CSV lub XLSX. Próbuje `openpyxl`, a w razie niepowodzenia używa fallback-parsingu XML/ZIP.

- `_custom_xlsx_to_df(xlsx_path: str) -> pd.DataFrame`  
  Parser awaryjny: rozpakowuje `.xlsx`, czyta `sharedStrings.xml` i pierwszy arkusz `xl/worksheets/*.xml`,
  samodzielnie rekonstruuje tabelę (wraz z nagłówkiem).

- `select_numeric_features(df: pd.DataFrame, target_col: str) -> list[str]`  
  Zwraca listę **numerycznych** kolumn cech (z wykluczeniem kolumny celu).
```python

from core.dataset_io import read_dataset, select_numeric_features
df = read_dataset("data.xlsx")
feats = select_numeric_features(df, target_col="Gallstone Status")

```

### 4.2. `core/preprocess.py`


**Funkcje:**

- `stratified_split(X, y, train_frac=0.7, seed=42)` → `(train_idx, test_idx)`  
  Dzieli wskaźniki próbek tak, by zachować proporcje klas 0/1.

- `zscore_fit(X)` → `(mu, sigma)`  
  Oblicza statystyki standaryzacji **tylko na train**.

- `zscore_transform(X, mu, sigma)` → `X'`  
  Zwraca znormalizowaną macierz cech. Zabezpiecza \( \sigma=0 \) → 1.0.
### 4.3. `core/metrics.py`


**Funkcje i metryki:**

- `safe_trapezoid(y, x)` – alias do `np.trapezoid` lub `np.trapz` (kompatybilność).
- `sigmoid(z)` – funkcja \( \sigma(z) = 1/(1+e^{-z}) \).
- `bce_loss(y_true, y_prob)` – Binary Cross-Entropy ze stabilnym `clip`.
- `accuracy(y_true, y_pred)` – odsetek trafień.
- `confusion(y_true, y_pred)` – zwraca *(tp, fp, fn, tn, tpr, fpr, prec, rec)*.
- `roc_curve_vals(y_true, y_score, num=200)` – generuje krzywą ROC.
- `pr_curve_vals(y_true, y_score, num=200)` – generuje krzywą PR.
- `binned_reliability(y_true, y_prob, bins=10)` – punkty do reliability diagram.
### 4.4. `core/platt.py`


**Kalibracja (Platt):**

- `fit_platt(y_true, y_score, max_iter=100, tol=1e-6)` → `(a, b)`  
  Uczy parametry \\(a,b\\) w regresji logistycznej \\( \sigma(a s + b) \\) metodą Newtona (zamknięty przypadek 2×2).

- `apply_platt(y_score, a, b)` → `p_calibrated`  
  Zwraca \\( \sigma(a s + b) \\).

**Złożoność:** liniowa w liczbie próbek; koszty per iteracja stałe (mała macierz 2×2).
### 4.5. `core/mlp.py` — MLPModel


**Klasa:** `MLPModel` (numpy-only, 1–2 warstwy ukryte)

**Inicjalizacja:**
- `n_in` – liczba cech wejściowych,
- `n_hidden=(h1,h2)` – rozmiary warstw (gdy `h2=0` → 1 warstwa),
- `lr, epochs, batch_size, activation ∈ {relu,tanh}, l2, seed`.

**Metody:**
- `forward(X)` → `(y, cache)`  
  Przepływ w przód; `y = sigmoid(z_out)`. `cache` niesie aktywacje do backprop.

- `backward(cache, y_true)` → `None`  
  Ręczny backprop z L2 dla wag; aktualizuje parametry SGD.

- `one_epoch(X, y)` → `None`  
  Losuje mini-batche, powtarza `forward+backward`.

- `predict_proba(X)` → `p ∈ (0,1)`  
  Zwraca prawdopodobieństwa klasy 1.

- `predict(X, threshold=0.5)` → etykiety {0,1}
### 4.6. `core/trainer.py` — Trainer


**Klasa:** `Trainer` — kapsułkuje pętlę treningową wraz z:
- early stopping (`use_es`, `patience`),
- step LR decay (`use_step_lr`, `step_every`, `gamma`),
- callbackami `on_epoch(ep, loss, val_loss, lr)` i `on_done(history)`,
- wsparciem dla flagi `stop_flag` (przerywanie treningu z GUI).

**Metoda:** `run()` – wykonuje pełny trening, zarządza najlepszymi wagami.
### 4.7. `core/analysis.py`


**Funkcje:**

- `make_test_table(Xte, yte, p_te, mu, sigma, features, test_idx=None)`  
  Buduje tabelę testową z oryginalną skalą cech, `y_true`, `p`, `y_pred`, `row_id`, `error_type` (FP/FN/OK).

- `permutation_importance(model, Xte, yte, features, base_probs=None, seed=123)`  
  Dla każdej cechy permutuje jej kolumnę i liczy spadek AUC: **ΔAUC = AUC_base − AUC_perm**.
### 4.8. `core/plots.py`


**Funkcje wykresów (zwracają `matplotlib.figure.Figure`):**
- `plot_loss(history)`,
- `plot_roc(y_true, p)`,
- `plot_pr(y_true, p)`,
- `plot_cm_metrics(y_true, p, thr)` – macierz pomyłek + słupki metryk.
## 5. API interfejsu (`gui/*`)


GUI jest oparte o **Tkinter/ttk**. Każda zakładka jest klasą dziedziczącą z `BaseTab`.
`BaseTab.draw_figure(...)` osadza wykres (matplotlib) z paskiem narzędzi.
### gui/base.py

**Metody/akcje:**

- `BaseTab.info()`
- `BaseTab.warn()`
- `BaseTab.error()`
- `BaseTab.draw_figure()`

### gui/data_tab.py

**Metody/akcje:**

- `DataTab.build()`
- `DataTab._browse()`
- `DataTab._load()`
- `DataTab._build_preview_table()`
- `DataTab._render_preview()`

### gui/train_tab.py

**Metody/akcje:**

- `TrainTab.build()`
- `TrainTab._prep()`
- `TrainTab._start()`
- `TrainTab._start_cv()`
- `TrainTab._stop()`
- `TrainTab._evaluate()`

### gui/predict_tab.py

**Metody/akcje:**

- `PredictTab.build()`
- `PredictTab._build_fields()`
- `PredictTab._predict()`

### gui/plots_tab.py

**Metody/akcje:**

- `PlotsTab.build()`
- `PlotsTab._loss()`
- `PlotsTab._roc()`
- `PlotsTab._pr()`
- `PlotsTab._cm()`

### gui/analysis_tab.py

**Metody/akcje:**

- `AnalysisTab.build()`
- `AnalysisTab._render_errors()`
- `AnalysisTab._importance()`
- `AnalysisTab._reliability()`

### gui/model_tab.py

**Metody/akcje:**

- `ModelTab.build()`
- `ModelTab._set_cal()`
- `ModelTab._save()`
- `ModelTab._load()`
- `ModelTab._learn_cal()`
- `ModelTab._batch()`

### gui/docs_tab.py

**Metody/akcje:**

- `DocsTab.build()`
- `DocsTab._search()`
- `DocsTab._copy_all()`
- `DocsTab._save_file()`

## 6. Przepływ pracy i najlepsze praktyki


1. **Przygotowanie danych**: upewnij się, że kolumna celu to 0/1, brakujące wartości są oczyszczone.
2. **Split & Z-score**: licz statystyki tylko na train, *nigdy* na całości.
3. **Hiperparametry**: zacznij od mniejszych LR (np. 1e-3), włącz ES, ewentualnie StepLR.
4. **Ocena**: oprócz accuracy patrz na ROC/AUC, PR/AUPRC; dopasuj **próg** do kosztów FP/FN.
5. **Kalibracja**: używaj **Platta** tylko na zbiorze walidacyjnym (w narzędziu – na teście dla prostoty).
6. **Eksport**: zapisuj model (wagi + μ,σ + lista cech + kalibracja).
7. **Batch predykcja**: dopilnuj tej samej kolejności i nazewnictwa kolumn cech.
## 7. Testowanie, walidacja, debugowanie


- **Jednostkowo**: sprawdź `zscore_fit/transform`, `roc_curve_vals`, `bce_loss` na prostych przypadkach.
- **Numerycznie**: porównaj gradienty z estymacją numeryczną na małej sieci (finite differences).
- **Losowość**: ustaw `seed` dla powtarzalności.
- **Wydajność**: zwiększ `batch_size`, zmniejsz `epochs` podczas prototypowania.
## 8. Częste problemy i rozwiązania


- **`ValueError: Brak kolumny celu`** – sprawdź nazwę `Gallstone Status`.
- **`NaN` w cechach** – wyczyść dane przed wczytaniem (lub imputuj).
- **Zerowa wariancja cechy** – w `zscore_fit` sigma==0 → 1.0 (bezpiecznie).
- **Overfitting** – zwiększ L2, włącz ES, użyj StepLR, zbieraj więcej danych.
- **Niestabilny trening** – zmniejsz LR, zwłaszcza dla ReLU; rozważ tanh.
## 9. Licencja i podziękowania


Kod bazuje na bibliotekach **numpy**, **pandas**, **matplotlib**, **Tkinter/ttk**.  
Możesz używać i modyfikować według własnych potrzeb w ramach swojego projektu.
### API: `core.dataset_io.read_dataset`

**Rodzaj:** function

**Sygnatura:** `read_dataset(path: str) -> pd.DataFrame`

**Opis:** Wczytuje dane z CSV/XLSX. Preferuje openpyxl, fallback na parser XML (.xlsx jako zip).

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.dataset_io import read_dataset
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.dataset_io._custom_xlsx_to_df`

**Rodzaj:** function

**Sygnatura:** `_custom_xlsx_to_df(xlsx_path: str) -> pd.DataFrame`

**Opis:** Awaryjny parser arkusza XLSX: sharedStrings + pierwszy worksheet, rozpoznawanie typów.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.dataset_io import _custom_xlsx_to_df
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.dataset_io.select_numeric_features`

**Rodzaj:** function

**Sygnatura:** `select_numeric_features(df: pd.DataFrame, target_col: str) -> list[str]`

**Opis:** Zwraca numeryczne kolumny cech z wykluczeniem kolumny celu.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.dataset_io import select_numeric_features
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.preprocess.stratified_split`

**Rodzaj:** function

**Sygnatura:** `stratified_split(X, y, train_frac=0.7, seed=42) -> (np.ndarray, np.ndarray)`

**Opis:** Dzieli indeksy na train/test zachowując proporcje klas.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.preprocess import stratified_split
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.preprocess.zscore_fit`

**Rodzaj:** function

**Sygnatura:** `zscore_fit(X) -> (mu: np.ndarray, sigma: np.ndarray)`

**Opis:** Liczy średnią i odchylenie standardowe (ddof=0); sigma==0 → 1.0.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.preprocess import zscore_fit
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.preprocess.zscore_transform`

**Rodzaj:** function

**Sygnatura:** `zscore_transform(X, mu, sigma) -> np.ndarray`

**Opis:** Z-score: (X - mu)/sigma, działa na macierzach.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.preprocess import zscore_transform
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.metrics.safe_trapezoid`

**Rodzaj:** function

**Sygnatura:** `safe_trapezoid(y, x) -> float`

**Opis:** Pole pod krzywą metodą trapezów; zgodność z różnymi NumPy.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.metrics import safe_trapezoid
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.metrics.sigmoid`

**Rodzaj:** function

**Sygnatura:** `sigmoid(z) -> np.ndarray`

**Opis:** Funkcja aktywacji wyjścia; numerycznie stabilna w typowym zakresie.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.metrics import sigmoid
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.metrics.bce_loss`

**Rodzaj:** function

**Sygnatura:** `bce_loss(y_true, y_prob, eps=1e-9) -> float`

**Opis:** Binary Cross-Entropy z przycięciem `y_prob` do [eps,1-eps].

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.metrics import bce_loss
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.metrics.accuracy`

**Rodzaj:** function

**Sygnatura:** `accuracy(y_true, y_pred) -> float`

**Opis:** Dokładność klasyfikacji.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.metrics import accuracy
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.metrics.confusion`

**Rodzaj:** function

**Sygnatura:** `confusion(y_true, y_pred) -> (tp, fp, fn, tn, tpr, fpr, prec, rec)`

**Opis:** Zwraca podstawowe składowe i metryki z macierzy pomyłek.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.metrics import confusion
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.metrics.roc_curve_vals`

**Rodzaj:** function

**Sygnatura:** `roc_curve_vals(y_true, y_score, num=200) -> (FPR, TPR)`

**Opis:** Generuje punkty ROC dla siatki progów [0,1].

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.metrics import roc_curve_vals
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.metrics.pr_curve_vals`

**Rodzaj:** function

**Sygnatura:** `pr_curve_vals(y_true, y_score, num=200) -> (REC, PREC)`

**Opis:** Generuje punkty PR (Recall, Precision).

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.metrics import pr_curve_vals
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.metrics.binned_reliability`

**Rodzaj:** function

**Sygnatura:** `binned_reliability(y_true, y_prob, bins=10) -> (mids, acc, cnt)`

**Opis:** Dane do wykresu kalibracji (reliability diagram).

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.metrics import binned_reliability
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.platt.fit_platt`

**Rodzaj:** function

**Sygnatura:** `fit_platt(y_true, y_score, max_iter=100, tol=1e-6) -> (a, b)`

**Opis:** Uczenie parametrów kalibracji Platta metodą Newtona (2x2).

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.platt import fit_platt
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.platt.apply_platt`

**Rodzaj:** function

**Sygnatura:** `apply_platt(y_score, a, b) -> np.ndarray`

**Opis:** Zastosowanie skalowania: sigma(a*s+b).

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.platt import apply_platt
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.mlp.MLPModel.__init__`

**Rodzaj:** method

**Sygnatura:** `MLPModel(n_in, n_hidden=(16,8), lr=1e-3, epochs=200, batch_size=32, activation='relu', l2=0.0, seed=42)`

**Opis:** Buduje model z inicjalizacją He/Xavier odpowiednio do architektury.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.mlp import MLPModel
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.mlp.MLPModel._act`

**Rodzaj:** method

**Sygnatura:** `_act(z) -> np.ndarray`

**Opis:** Zwraca ReLU lub tanh w zależności od ustawienia.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.mlp import MLPModel
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.mlp.MLPModel._act_grad`

**Rodzaj:** method

**Sygnatura:** `_act_grad(z) -> np.ndarray`

**Opis:** Pochodna ReLU/tanh dla backprop.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.mlp import MLPModel
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.mlp.MLPModel.forward`

**Rodzaj:** method

**Sygnatura:** `forward(X) -> (y, cache)`

**Opis:** Przepływ w przód; cache zawiera aktywacje na potrzeby backward.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.mlp import MLPModel
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.mlp.MLPModel.backward`

**Rodzaj:** method

**Sygnatura:** `backward(cache, y_true) -> None`

**Opis:** Backprop + aktualizacja wag (mini-batch SGD) z L2.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.mlp import MLPModel
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.mlp.MLPModel.one_epoch`

**Rodzaj:** method

**Sygnatura:** `one_epoch(X, y) -> None`

**Opis:** Losowanie indeksów; pętle po batchach; forward/backward.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.mlp import MLPModel
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.mlp.MLPModel.predict_proba`

**Rodzaj:** method

**Sygnatura:** `predict_proba(X) -> np.ndarray`

**Opis:** Zwraca wektor prawdopodobieństw klasy pozytywnej.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.mlp import MLPModel
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.mlp.MLPModel.predict`

**Rodzaj:** method

**Sygnatura:** `predict(X, threshold=0.5) -> np.ndarray[int]`

**Opis:** Zwraca predykcję etykiet przy zadanym progu.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.mlp import MLPModel
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.trainer.Trainer.__init__`

**Rodzaj:** method

**Sygnatura:** `Trainer(model, Xtr, ytr, Xte=None, yte=None, use_es=True, patience=20, use_step_lr=False, step_every=50, gamma=0.5, on_epoch=None, on_done=None, stop_flag=None)`

**Opis:** Inicjalizacja trenera z callbackami i opcjami harmonogramu/ES.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.trainer import Trainer
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.trainer.Trainer.run`

**Rodzaj:** method

**Sygnatura:** `run() -> None`

**Opis:** Główna pętla treningowa + zarządzanie najlepszymi wagami.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.trainer import Trainer
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.analysis.make_test_table`

**Rodzaj:** function

**Sygnatura:** `make_test_table(Xte, yte, p_te, mu, sigma, features, test_idx=None) -> pd.DataFrame`

**Opis:** Tabela testowa z etykietami, predykcją, progiem i typem błędu.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.analysis import make_test_table
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.analysis.permutation_importance`

**Rodzaj:** function

**Sygnatura:** `permutation_importance(model, Xte, yte, features, base_probs=None, seed=123) -> list[(feature, delta_auc)]`

**Opis:** Spadek AUC po permutacji kolumny — miara ważności.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.analysis import permutation_importance
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.plots.plot_loss`

**Rodzaj:** function

**Sygnatura:** `plot_loss(history) -> Figure`

**Opis:** Wykres strat (train/val).

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.plots import plot_loss
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.plots.plot_roc`

**Rodzaj:** function

**Sygnatura:** `plot_roc(y_true, p) -> Figure`

**Opis:** Wykres ROC z AUC.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.plots import plot_roc
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.plots.plot_pr`

**Rodzaj:** function

**Sygnatura:** `plot_pr(y_true, p) -> Figure`

**Opis:** Wykres Precision–Recall z AUPRC.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.plots import plot_pr
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### API: `core.plots.plot_cm_metrics`

**Rodzaj:** function

**Sygnatura:** `plot_cm_metrics(y_true, p, thr) -> Figure`

**Opis:** Macierz pomyłek + słupki metryk dla danego progu.

#### Parametry i zwracane wartości

- Patrz sygnatura oraz opisy powyżej.

#### Złożoność obliczeniowa

- Zależna od rozmiarów wejść. Operacje wektorowe NumPy.

#### Przykład użycia

```python
from core.plots import plot_cm_metrics
# ... uzupełnij własnymi danymi ...
```

#### Błędy i pułapki

- Sprawdź typy i kształty macierzy (`shape`).
- Upewnij się, że kolumna celu to 0/1 gdzie wymagane.

### Backprop — szkic wyprowadzenia


Dla warstwy \\( l \\): \\( z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)} \\), \\( a^{(l)} = \phi(z^{(l)}) \\).
Dla wyjścia binarnego: \\( \hat{y} = \sigma(z^{(L)}) \\), strata BCE.  
Błąd na wyjściu: \\( \delta^{(L)} = \hat{y} - y \\).  
Dla warstw ukrytych: \\( \delta^{(l)} = (\delta^{(l+1)} {W^{(l+1)}}^\top) \odot \phi'(z^{(l)}) \\).  
Gradienty: \\( \frac{\partial \mathcal{L}}{\partial W^{(l)}} = {a^{(l-1)}}^\top \delta^{(l)} + \lambda W^{(l)} \\), \\( \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \sum \delta^{(l)} \\).

### Step LR Decay — kiedy używać?


- Gdy obserwujesz płaskowyż strat: zmniejszenie LR może pomóc zejść do głębszego minimum.
- Zbyt agresywne schodki (małe `step_every`, bardzo małe `gamma`) wydłużają trening.

### K-fold CV — wariancja i uśrednianie


- Używaj stratyfikacji dla klasy binarnej, aby nie uzyskać foldów bez klasy 1.
- Raportuj średnią ± odchylenie standardowe dla kluczowych metryk (AUC, ACC).

### Permutation Importance — interpretacja


- Duża ΔAUC po permutacji cechy → cecha istotna dla dyskryminacji modelu.
- Uwaga: zależna od korelacji między cechami; nie jest to „przyczynowość”.

### Kalibracja — praktyka


- Ucz na walidacji, zastosuj na teście; unikaj uczenia kalibracji na tych samych danych raportowych.
- Sprawdź reliability diagram przed i po kalibracji.

## FAQ (krótkie odpowiedzi)

**P1.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P2.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P3.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P4.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P5.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P6.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P7.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P8.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P9.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P10.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P11.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P12.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P13.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P14.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P15.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P16.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P17.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P18.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P19.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P20.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P21.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P22.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P23.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P24.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P25.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P26.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P27.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P28.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P29.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P30.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P31.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P32.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P33.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P34.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P35.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P36.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P37.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P38.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P39.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P40.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P41.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P42.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P43.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P44.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P45.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P46.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P47.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P48.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P49.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.
**P50.** Jak dobrać próg decyzyjny? — Użyj suwaka w CM i kieruj się kosztami FP/FN oraz metrykami PR/ROC.

\n'''