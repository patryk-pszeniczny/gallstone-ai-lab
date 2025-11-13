DOCS_TEXT = r'''\
# Gallstone AI Lab – dokumentacja techniczna

## 1. Cel projektu i kontekst medyczny

Projekt **Gallstone AI Lab** służy do **wspomagania oceny ryzyka kamicy pęcherzyka żółciowego (cholelithiasis)** na podstawie danych klinicznych pacjenta. Aplikacja **nie jest wyrobem medycznym** – ma charakter **edukacyjny / dydaktyczny** i została przygotowana jako przykład:

- kompletnego projektu w Pythonie z wyraźnym podziałem na logikę i GUI,
- użycia klasycznego modelu uczenia maszynowego (MLP – *Multi-Layer Perceptron*),
- użycia alternatywnego, **wyjaśnialnego** systemu opartego o **logikę rozmytą** (fuzzy logic),
- pracy na realnym, opublikowanym zbiorze danych medycznych (*Gallstone Dataset – UCI*).

Główny cel programu:

> Na podstawie cech klinicznych pacjenta oszacować prawdopodobieństwo występowania kamicy pęcherzyka żółciowego oraz porównać wynik:
> - klasycznego modelu MLP,
> - silnika opartego o logikę rozmytą.

Wynik ma formę:
- **etykiety binarnej** (np. „wysokie prawdopodobieństwo kamicy” vs „niskie prawdopodobieństwo”) oraz
- **prawdopodobieństwa numerycznego / stopnia przynależności** w przedziale \([0,1]\).

---

## 2. Zbiór danych: Gallstone (UCI)

### 2.1. Źródło danych

Dane pochodzą ze zbioru **Gallstone** opublikowanego w **UCI Machine Learning Repository** (ID: 1150). Zbiór został przygotowany na podstawie danych pacjentów z **VM Medical Park Hospital w Ankarze (Turcja)**.

Charakterystyka zbioru (wg opisu UCI):

- liczba pacjentów: ok. **319 osób** (część materiałów podaje 320 instancji),
- z tego **161 osób** z rozpoznaną kamicą żółciową,
- liczba cech (features): **~37 cech wejściowych**,
- jedna zmienna docelowa (target): **Gallstone Status** – informacja, czy kamica jest obecna,
- dane są **kompletne (brak braków danych)** i mają zbalansowane klasy (z grubsza tyle samo chorych i zdrowych).

Zbiór obejmuje:
- **cechy demograficzne**,
- **cechy bioimpedancyjne** (kompozycja ciała),
- **parametry laboratoryjne** (biochemia, funkcja wątroby, nerek, stan zapalny).

W projekcie dane te znajdują się w pliku:

- `dataset-uci.xlsx` – lokalna kopia zestawu Gallstone (UCI) w formacie Excela.

### 2.2. Zmienna docelowa (target)

Zmienną wyjściową (klasą) jest:

- **Gallstone Status** – zmienna binarna:
  - `0` – *kamica obecna (pacjent z kamicą)*,
  - `1` – *kamica nieobecna (osoba zdrowa / kontrolna)*.

W modelu MLP i w systemie fuzzy jest to wartość, którą próbujemy przewidzieć.

### 2.3. Główne grupy cech wejściowych

Poniżej zebrano najważniejsze grupy cech występujących w zbiorze Gallstone (UCI). Nazwy w nawiasach odpowiadają oryginalnym kolumnom.

#### 2.3.1. Cechy demograficzne

- **Wiek (Age)** – wiek pacjenta w latach.
- **Płeć (Gender)** – płeć (zakodowana binarnie, np. 0 – mężczyzna, 1 – kobieta).
- **Wzrost (Height)** – wzrost w centymetrach.
- **Masa ciała (Weight)** – masa ciała w kilogramach.
- **BMI (Body Mass Index)** – wskaźnik masy ciała:

  \[
  \text{BMI} = \frac{\text{masa [kg]}}{\text{wzrost [m]}^2}
  \]

#### 2.3.2. Choroby współistniejące

- **Comorbidity** – liczba / skala chorób współistniejących (0 – brak, 1 – jedna, 2 – dwie, 3 – ≥3).
- **Coronary Artery Disease (CAD)** – choroba wieńcowa (0 – nie, 1 – tak).
- **Hypothyroidism** – niedoczynność tarczycy (0 – nie, 1 – tak).
- **Hyperlipidemia** – hipercholesterolemia / nieprawidłowy profil lipidowy (0 – nie, 1 – tak).
- **Diabetes Mellitus (DM)** – cukrzyca (0 – nie, 1 – tak).

#### 2.3.3. Parametry bioimpedancji i składu ciała

- **Total Body Water (TBW)** – całkowita woda w organizmie.
- **Extracellular Water (ECW)** – woda pozakomórkowa.
- **Intracellular Water (ICW)** – woda wewnątrzkomórkowa.
- **ECF/TBW** – udział wody pozakomórkowej w całkowitej wodzie.
- **Total Body Fat Ratio (TBFR)** – procent tkanki tłuszczowej w organizmie.
- **Lean Mass (LM)** – masa beztłuszczowa.
- **Protein** – szacunkowa ilość białka w organizmie.
- **Visceral Fat Rating (VFR)** – skala otłuszczenia trzewnego.
- **Bone Mass (BM)** – masa kostna.
- **Muscle Mass (MM)** – masa mięśniowa.
- **Obesity** – wskaźnik otyłości (procent nadmiernej tkanki tłuszczowej).
- **Total Fat Content (TFC)** – całkowita ilość tłuszczu.
- **Visceral Fat Area (VFA)** – powierzchnia trzewnej tkanki tłuszczowej.
- **Visceral Muscle Area (VMA)** – powierzchnia mięśni w obrębie trzewi.
- **Hepatic Fat Accumulation (HFA)** – stłuszczenie wątroby (kategorycznie, np. 0–4).

#### 2.3.4. Parametry laboratoryjne

- **Glucose** – glikemia (stężenie glukozy).
- **Total Cholesterol (TC)** – cholesterol całkowity.
- **LDL (Low Density Lipoprotein)** – „zły” cholesterol.
- **HDL (High Density Lipoprotein)** – „dobry” cholesterol.
- **Triglyceride** – triglicerydy.
- **AST (Aspartate Aminotransferase)** – enzym wątrobowy.
- **ALT (Alanine Aminotransferase)** – enzym wątrobowy.
- **ALP (Alkaline Phosphatase)** – enzym wątrobowo‑kostny.
- **Creatinine** – parametr czynności nerek.
- **GFR (Glomerular Filtration Rate)** – wskaźnik filtracji kłębuszkowej.
- **CRP (C-Reactive Protein)** – białko ostrej fazy (stan zapalny).
- **HGB (Hemoglobin)** – hemoglobina (transport tlenu we krwi).
- **Vitamin D** – stężenie witaminy D.

### 2.4. Przygotowanie danych w projekcie

Logika projektu zakłada typowy pipeline:

1. **Wczytanie danych** z pliku `dataset-uci.xlsx` (np. biblioteka `pandas`).
2. **Oddzielenie cech wejściowych X i celu y**:
   - \(X \in \mathbb{R}^{n \times d}\) – macierz cech (n – liczba pacjentów, d – liczba cech),
   - \(y \in \{0,1\}^n\) – etykiety Gallstone Status.
3. **Podział na zbiór treningowy i testowy**, np. 70% / 30% lub 80% / 20%.
4. **Standaryzacja / skalowanie cech ciągłych**, np.:

   \[
   x' = \frac{x - \mu}{\sigma}
   \]

   gdzie:
   - \(\mu\) – średnia z danej cechy w zbiorze treningowym,
   - \(\sigma\) – odchylenie standardowe.

5. Ewentualne **kodowanie cech kategorycznych** (jeśli jeszcze nie są zakodowane) – w dataset‑cie Gallstone większość jest już gotowa (0,1,2,…).

---

## 3. Model MLP (Multi-Layer Perceptron)

### 3.1. Idea MLP

Perceptron wielowarstwowy (MLP) to klasyczna sieć neuronowa:

- przyjmuje wektor cech wejściowych \(x \in \mathbb{R}^d\),
- przeprowadza go przez jedną lub więcej **warstw ukrytych**,
- na wyjściu zwraca prawdopodobieństwo przynależności do klasy „kamica obecna”.

W projekcie wykorzystano styl pracy znany z biblioteki **scikit-learn** (klasa `MLPClassifier` lub podobna), co umożliwia:

- prostą konfigurację liczby warstw i neuronów,
- wybór funkcji aktywacji,
- automatyczną optymalizację wag.

### 3.2. Struktura sieci

Dla przypadku zbioru Gallstone:

- liczba wejść: \(d\) – liczba cech (ok. 37),
- wyjście: jedna neuron‑jednostka z aktywacją sigmoidalną lub softmax dla klasy binarnej.

Abstrakcyjnie:

- **warstwa wejściowa**: \(a^{(0)} = x\),
- **warstwy ukryte**: dla każdej warstwy \(l = 1, 2, \dots, L - 1\):

  \[
  z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad
  a^{(l)} = f^{(l)}(z^{(l)})
  \]

  gdzie:
  - \(W^{(l)}\) – macierz wag,
  - \(b^{(l)}\) – wektor biasów,
  - \(f^{(l)}\) – nieliniowa funkcja aktywacji (np. ReLU, tanh).

- **warstwa wyjściowa**:

  \[
  z^{(L)} = W^{(L)} a^{(L-1)} + b^{(L)}, \quad
  \hat{y} = \sigma(z^{(L)})
  \]

  gdzie \(\sigma\) to funkcja sigmoidalna:

  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]

Wynik \(\hat{y} \in (0,1)\) interpretuje się jako **oszacowane prawdopodobieństwo występowania kamicy**.

### 3.3. Funkcja straty – log-loss (binary cross-entropy)

Dla pojedynczego przykładu \((x_i, y_i)\), gdzie \(y_i \in \{0,1\}\) i \(\hat{y}_i\) to wyjście sieci, funkcja straty ma postać:

\[
\mathcal{L}_i = -\left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]
\]

Dla całego zbioru uczącego (N przykładów):

\[
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_i
\]

Minimalizacja tej funkcji powoduje, że sieć uczy się przypisywać wysokie prawdopodobieństwa poprawnej klasie.

### 3.4. Uczenie sieci – propagacja wsteczna

Optymalizacja wag odbywa się za pomocą wariantu **spadku gradientowego**, często z momentem lub innymi ulepszeniami (np. Adam). Podstawowa idea:

1. **Propagacja w przód** – obliczamy \(\hat{y}_i\) dla danego przykładu.
2. **Obliczenie błędu** – liczymy \(\mathcal{L}_i\).
3. **Propagacja wsteczna (backpropagation)** – liczymy pochodne \(\frac{\partial \mathcal{L}}{\partial W^{(l)}}\), \(\frac{\partial \mathcal{L}}{\partial b^{(l)}}\).
4. **Aktualizacja wag**:

   \[
   W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(l)}}, \quad
   b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial b^{(l)}}
   \]

   gdzie \(\eta\) to współczynnik uczenia (learning rate).

W praktycznej implementacji w scikit‑learn szczegóły backpropagacji i optymalizacji są ukryte za interfejsem `fit()`.

### 3.5. Metryki oceny jakości modelu

Po wytrenowaniu modelu MLP oceniamy go na zbiorze testowym. Dla klasyfikacji binarnej zwykle liczy się:

- **macierz pomyłek** (confusion matrix):

  - TP – *True Positives* (poprawnie zaklasyfikowani pacjenci z kamicą),
  - TN – *True Negatives* (poprawnie zaklasyfikowani zdrowi),
  - FP – *False Positives* (fałszywie „chory”),
  - FN – *False Negatives* (fałszywie „zdrowy”).

Na tej podstawie obliczamy:

- **Accuracy (dokładność)**:

  \[
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  \]

- **Precision (precyzja)** – wśród przewidzianych „chorych” ilu jest faktycznie chorych:

  \[
  \text{Precision} = \frac{TP}{TP + FP}
  \]

- **Recall (czułość, TPR)** – jaki odsetek realnie chorych wykrył model:

  \[
  \text{Recall} = \frac{TP}{TP + FN}
  \]

- **F1-score** – średnia harmoniczna precyzji i czułości:

  \[
  F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  \]

Dodatkowo można analizować krzywą ROC oraz AUC.

---

## 4. Silnik oparty o logikę rozmytą (fuzzy logic)

### 4.1. Dlaczego logika rozmyta?

Modele oparte na logice rozmytej są:

- bardziej **interpretowalne** – można zapisać reguły w stylu „IF… THEN…”,
- zbliżone do sposobu, w jaki **lekarz** rozumuje („pacjent starszy, z otyłością i wysokim cholesterolem → wysokie ryzyko kamicy”),
- odporne na pewną niepewność i nieostrość granic (np. „wysokie BMI”, „średni poziom cholesterolu”).

W projekcie silnik fuzzy stanowi **alternatywę** wobec MLP i umożliwia porównanie:

- „czarnej skrzynki” (MLP),
- „białej skrzynki” (zestaw reguł rozmytych).

### 4.2. Zmienne lingwistyczne

Dla wybranych cech ze zbioru Gallstone definiujemy **zmienne lingwistyczne**, np.:

- WIEK – {młody, średni, starszy},
- BMI – {prawidłowe, nadwaga, otyłość},
- HFA (stłuszczenie wątroby) – {brak, łagodne, umiarkowane, ciężkie},
- TC (cholesterol całkowity) – {niski, prawidłowy, wysoki},
- GFR – {prawidłowa czynność nerek, obniżona},
- VFR – {niski tłuszcz trzewny, umiarkowany, wysoki},
- itp.

Każda z tych wartości lingwistycznych odpowiada **zbiorowi rozmytemu** opisującemu pewien przedział liczbowy.

### 4.3. Funkcje przynależności

Zbiory rozmyte opisujemy funkcjami przynależności \(\mu_A(x) \in [0,1]\). Przykładowo, dla trójkątnej funkcji przynależności:

- parametry: \(a < b < c\),
- \(a\) – punkt, w którym zaczyna się rosnąca część,
- \(b\) – środek (wartość 1),
- \(c\) – punkt, w którym funkcja opada do 0.

Definicja:

\[
\mu_A(x) =
\begin{cases}
0, x \le a \\
\dfrac{x - a}{b - a}, a < x \le b \\
\dfrac{c - x}{c - b}, b < x < c \\
0, x \ge c
\end{cases}
\]


Analogicznie można stosować funkcje trapezowe, gaussowskie itd. W projekcie stosuje się proste **funkcje trójkątne / trapezowe**, które łatwo dobrać i interpretować.

### 4.4. Reguły rozmyte (Mamdani‑style)

Reguły w systemie są zapisane w postaci:

> \(R_j\): JEŻELI (WIEK jest *starszy*) ORAZ (BMI jest *otyłość*) ORAZ (HFA jest *umiarkowane lub ciężkie*)  
> TO (RYZYKO\_KAMICY jest *wysokie*).

Formalnie, dla danej reguły \(R_j\):

\[
R_j: \text{JEŻELI } A_{1}^{(j)} \wedge A_{2}^{(j)} \wedge \dots \wedge A_{k}^{(j)} \text{ TO } B^{(j)}
\]

gdzie:
- \(A_{i}^{(j)}\) – przesłanki (zbiory rozmyte na wejściu),
- \(B^{(j)}\) – konkluzja (zbiór rozmyty na wyjściu, np. „niskie”, „średnie”, „wysokie ryzyko”).

### 4.5. Wnioskowanie rozmyte

Dla zadanego pacjenta:

1. **Fuzzification** – zamiana wartości liczbowych na stopnie przynależności do zbiorów rozmytych.
2. **Obliczenie stopnia spełnienia reguł** – typowo:

   - dla operatora AND stosuje się:

     \[
     \mu_{A \wedge B}(x) = \min\left(\mu_A(x), \mu_B(x)\right)
     \]

   - dla operatora OR:

     \[
     \mu_{A \vee B}(x) = \max\left(\mu_A(x), \mu_B(x)\right)
     \]

3. **Agregacja wniosków** – łączymy wyniki wszystkich reguł w jeden zbiór rozmyty reprezentujący zmienną wyjściową (np. *RYZYKO\_KAMICY*).
4. **Defuzyfikacja** – przekształcamy zbiór rozmyty na wartość liczbową, np. metoda środka ciężkości (center of gravity):

   \[
   y^\* = \frac{\int y \, \mu_R(y) \, dy}{\int \mu_R(y) \, dy}
   \]

   gdzie \(\mu_R(y)\) to wynikowa funkcja przynależności dla zmiennej wyjściowej (np. ryzyko).

Wynik \(y^\*\) jest następnie skalowany do przedziału \([0,1]\) lub do skali procentowej, a także może być przemapowany na etykietę:

- \(y^\* < 0.33\) → niskie ryzyko,
- \(0.33 \le y^\* < 0.66\) → średnie ryzyko,
- \(y^\* \ge 0.66\) → wysokie ryzyko.

(Granice progów można modyfikować w kodzie w zależności od założeń.)

### 4.6. Eksport wyników do CSV

Silnik fuzzy w projekcie dodatkowo umożliwia **eksport wyników** np. do pliku CSV:

- dla każdego pacjenta zapisywane są:
  - identyfikator / indeks,
  - wartość wyjściowa fuzzy (np. ryzyko w [0,1]),
  - zaklasyfikowana klasa (np. 0/1 lub niskie/średnie/wysokie).

Przykładowy plik wynikowy: `gallstone_fuzzy_predictions.csv`.

---

## 5. Architektura projektu

Struktura katalogów (zgodnie z opisem w README):

- `main.py` – **główny plik uruchomieniowy**:
  - inicjalizuje logikę,
  - ładuje / trenuje modele,
  - uruchamia warstwę GUI.
- `core/` – **warstwa logiki biznesowej i uczenia maszynowego**:
  - wczytywanie danych,
  - przygotowanie cech,
  - trenowanie i zapis modelu MLP,
  - implementacja silnika fuzzy,
  - obliczanie metryk i ewaluacja.
- `gui/` – **warstwa interfejsu użytkownika**:
  - formularze do wprowadzania cech pacjenta,
  - przyciski wywołujące predykcję MLP i fuzzy,
  - wyświetlanie wyników,
  - ewentualne opcje eksportu / ustawień.
- `assets/` – zasoby graficzne na potrzeby GUI (ikony, logotypy itp.).
- dane:
  - `dataset-uci.xlsx` – wejściowy zbiór danych (Gallstone – UCI),
  - `gallstone_fuzzy_predictions.csv` – przykładowy plik wyników fuzzy.

Logika aplikacji jest rozdzielona tak, aby:

- łatwo było podmienić model MLP (np. na inny klasyfikator),
- można było rozbudowywać zestaw reguł fuzzy bez ingerencji w GUI,
- GUI pozostawało cienką warstwą prezentacji.

---

## 6. Przepływ działania programu

Ogólny scenariusz:

1. **Start programu**:
   - użytkownik uruchamia `python main.py`,
   - inicjalizowane jest środowisko (wczytanie modeli / ewentualny trening).

2. **Wczytanie / trenowanie modelu MLP**:
   - jeśli istnieje zapisany model (np. w pliku), aplikacja może go wczytać,
   - w przeciwnym razie:
     - wczytywany jest `dataset-uci.xlsx`,
     - dane są dzielone na zbiór treningowy i testowy,
     - trenowany jest `MLPClassifier`,
     - wynikowy model może być zapisywany na dysku.

3. **Przygotowanie silnika fuzzy**:
   - definiowane są zmienne lingwistyczne,
   - przypisywane funkcje przynależności (parametry „a, b, c” dla funkcji trójkątnych itp.),
   - ładowany jest zbiór reguł (`if–then`).

4. **Uruchomienie GUI**:
   - użytkownik wybiera tryb pracy:
     - pojedynczy pacjent – ręczne wprowadzenie danych,
     - ewaluacja na całym zbiorze (np. do porównania metryk),
   - po wciśnięciu przycisku **„Oblicz”** aplikacja:
     - przekształca dane wejściowe do odpowiedniego formatu,
     - przekazuje je do MLP i/lub silnika fuzzy,
     - wyświetla wynik (predykcja MLP i wynik fuzzy).

5. **Wyświetlanie i zapis rezultatów**:
   - w GUI pokazuje się:
     - przewidywana klasa (0/1),
     - prawdopodobieństwo z MLP,
     - wynik fuzzy (np. wartość z przedziału [0,1] i opis słowny),
   - opcjonalnie użytkownik może:
     - zapisać wyniki do pliku CSV,
     - porównać metryki modeli.

---

## 7. Opis interfejsu użytkownika (GUI)

Szczegółowa implementacja GUI zależy od użytej biblioteki (Tkinter / PyQt / inna). Niezależnie od narzędzia, logika jest podobna.

Typowe elementy interfejsu:

1. **Formularz danych pacjenta**:
   - pola tekstowe / numeryczne dla:
     - wieku, wzrostu, masy ciała, BMI,
     - chorób współistniejących (CAD, DM, itd.),
     - parametrów bioimpedancji,
     - parametrów laboratoryjnych (glukoza, cholesterol, enzymy wątrobowe, GFR, CRP, HGB, Vit. D),
   - listy rozwijane dla cech kategorycznych (np. HFA 0–4).

2. **Wybór modelu**:
   - przełącznik / checkbox:
     - [ ] użyj modelu MLP,
     - [ ] użyj silnika fuzzy,
     - [ ] użyj obu i porównaj wyniki.

3. **Przycisk obliczenia**:
   - przycisk „**Oblicz predykcję**” / „**Analizuj**”,
   - po kliknięciu:
     - dane są walidowane,
     - aplikacja wywołuje odpowiednie metody w warstwie `core`.

4. **Panel wyników**:
   - tekstowy opis predykcji MLP:
     - np. „Prawdopodobieństwo kamicy wg MLP: 0.78 (78%) – klasa: kamica obecna”.
   - opis wyniku fuzzy:
     - np. „Silnik fuzzy: ryzyko 0.65 (wysokie) – klasa: kamica obecna”.
   - ewentualne dodatkowe informacje:
     - interpretacja (np. „podwyższone BMI, istotne stłuszczenie wątroby i nieprawidłowy profil lipidowy”).

5. **Przyciski dodatkowe**:
   - np. „Zapisz wyniki do CSV”,
   - „Pokaż dokumentację” – może otwierać tekst z `docs_text.py` w osobnym oknie.

---

## 8. Uruchamianie i środowisko

Przykładowa procedura (zgodna z README):

1. Klonowanie repozytorium:

   ```bash
   git clone https://github.com/patryk-pszeniczny/gallstone-ai-lab.git
   cd gallstone-ai-lab
   ```

2. Utworzenie i aktywacja wirtualnego środowiska (Python 3.x):

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

3. Instalacja zależności:

   ```bash
   pip install -r requirements.txt
   ```

4. Uruchomienie aplikacji:

   ```bash
   python main.py
   ```

Po uruchomieniu aplikacja wczyta dane, przygotuje modele i otworzy GUI.

---

## 9. Ograniczenia projektu

1. **Nie jest to wyrób medyczny** – model ma charakter demonstracyjny.
2. **Rozmiar zbioru danych** – 319 osób to relatywnie mały zbiór jak na modele ML, co może:
   - sprzyjać przeuczeniu,
   - ograniczać uogólnialność wyników.
3. **Źródło danych z jednego ośrodka** – pacjenci pochodzą z jednego szpitala (Turcja), więc:
   - rozkład cech populacji może różnić się od innych krajów / grup etnicznych.
4. **Brak pełnej walidacji klinicznej** – projekt nie zastępuje badań klinicznych ani wytycznych towarzystw naukowych.
5. **Dobór reguł fuzzy** – reguły zostały zdefiniowane „ekspercko” lub pół‑heurystycznie, a nie na podstawie formalnych badań klinicznych.

---

## 10. Możliwe rozszerzenia

1. **Rozbudowa modelu MLP**:
   - strojenie hiperparametrów (liczba warstw, neuronów, współczynnika uczenia),
   - porównanie z innymi modelami (Random Forest, XGBoost, SVM).

2. **Zaawansowana interpretowalność**:
   - wykorzystanie SHAP / LIME do wyjaśniania predykcji MLP,
   - wizualizacja wpływu poszczególnych cech na wynik.

3. **Rozbudowa silnika fuzzy**:
   - dodanie większej liczby reguł,
   - automatyczna identyfikacja reguł (np. na podstawie danych),
   - dynamiczna regulacja funkcji przynależności.

4. **Walidacja krzyżowa i raporty**:
   - k-fold cross‑validation,
   - generowanie raportów (np. PDF) z wynikami, krzywymi ROC itd.

5. **Integracja z innymi danymi**:
   - połączenie z danymi obrazowymi (USG),
   - integracja danych z innych ośrodków.

---

## 11. Podsumowanie

Projekt **Gallstone AI Lab** prezentuje kompletną ścieżkę:

1. Praca na rzeczywistym zbiorze klinicznym (Gallstone – UCI).
2. Zastosowanie klasycznego modelu uczenia maszynowego (MLP).
3. Zastosowanie alternatywnego, interpretowalnego modelu fuzzy.
4. Integracja wszystkiego w jedną, prostą w obsłudze aplikację z GUI.

Dzięki temu projekt może być wykorzystywany:

- jako **materiał na zajęcia uczelniane** (systemy ekspertowe, logika rozmyta, podstawy AI w medycynie),
- jako **pokazowy projekt** na prezentacje / portfolio,
- jako punkt wyjścia do dalszych badań i rozszerzeń w kierunku medycznych systemów wspomagania decyzji (CDSS – *Clinical Decision Support Systems*).

Należy zawsze pamiętać, że **ostateczna decyzja medyczna należy do lekarza**, a tego typu narzędzia mają jedynie charakter pomocniczy i edukacyjny.\n
'''