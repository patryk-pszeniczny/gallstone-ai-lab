# app_fuzzy_gallstone.py
# ============================================
# Tkinter + FuzzyLogic (Mamdani-lite) do GallStone
# - Tryb "Naucz" (fit): uczy MF z datasetu i próg (Youden)
# - Tryb "Predykcja": formularz -> wynik + decyzja + wyjaśnienie
# - Zapis/Wczytanie modelu (JSON)
# Bez ML frameworków (torch/sklearn) – tylko numpy/pandas/matplotlib
# ============================================

import os
import io
import json
import zipfile
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# 0) Utils: bezpieczny AUC
# ==============================
def safe_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

# ==============================
# 1) Wczytywanie danych (z fallbackiem na custom XLSX parser)
# ==============================
def try_read_excel(path):
    try:
        df = pd.read_excel(path)
        if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
            return df
    except Exception:
        pass
    return custom_xlsx_to_df(path)

def custom_xlsx_to_df(xlsx_path):
    def strip_tag(tag):
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    import xml.etree.ElementTree as ET
    z = zipfile.ZipFile(xlsx_path)

    shared_strings = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = z.read("xl/sharedStrings.xml").decode("utf-8", errors="ignore")
        root = ET.fromstring(ss)
        for si in root:
            texts = []
            for elem in si.iter():
                if strip_tag(elem.tag) == "t" and elem.text is not None:
                    texts.append(elem.text)
            shared_strings.append("".join(texts))

    sheet_name = None
    for n in z.namelist():
        if n.startswith("xl/worksheets/") and n.endswith(".xml"):
            sheet_name = n
            break
    if sheet_name is None:
        raise RuntimeError("Brak arkusza w XLSX.")

    xml_data = z.read(sheet_name).decode("utf-8", errors="ignore")
    root = ET.fromstring(xml_data)

    rows_data = {}
    max_col = 0

    for row in root.iter():
        if strip_tag(row.tag) == "row":
            r_idx = int(row.attrib.get("r", "0"))
            cells = {}
            for c in row:
                if strip_tag(c.tag) != "c":
                    continue
                ref = c.attrib.get("r")  # A1, B2...
                col_letters = "".join(ch for ch in ref if ch.isalpha()) if ref else None
                col_idx = 0
                if col_letters:
                    for ch in col_letters:
                        col_idx = col_idx*26 + (ord(ch.upper()) - ord('A') + 1)
                t = c.attrib.get("t")  # 's' = shared string
                v_text = None
                for child in c:
                    if strip_tag(child.tag) == "v":
                        v_text = child.text
                        break
                if t == "s" and v_text is not None:
                    si = int(v_text)
                    val = shared_strings[si] if 0 <= si < len(shared_strings) else ""
                else:
                    val = v_text
                if val is not None:
                    try:
                        if isinstance(val, str) and ('.' in val or 'e' in val.lower()):
                            val = float(val)
                        else:
                            val = int(val)
                    except Exception:
                        pass
                cells[col_idx] = val
                max_col = max(max_col, col_idx)
            rows_data[r_idx] = cells

    row_indices = sorted(rows_data.keys())
    table = []
    for r in row_indices:
        row = [rows_data[r].get(c, None) for c in range(1, max_col+1)]
        table.append(row)

    header = None
    data_start = 0
    for i, row in enumerate(table):
        if any(x is not None for x in row):
            header = [str(x).strip() if x is not None else "" for x in row]
            data_start = i + 1
            break

    data = table[data_start:]
    df = pd.DataFrame(data, columns=header)
    return df

# ==============================
# 2) FuzzyLogic: MF, reguły, scoring, ewaluacja
# ==============================
def tri(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / max(b - a, 1e-9)
    return (c - x) / max(c - b, 1e-9)

def trap(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / max(b - a, 1e-9)
    return (d - x) / max(d - c, 1e-9)

def build_memberships(df, feature_cols):
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    q = df[numeric_features].quantile([0.15, 0.5, 0.85]).T
    q.columns = ["q15", "q50", "q85"]

    mf_params = {}
    for c in numeric_features:
        aL, bM, dH = q.loc[c, "q15"], q.loc[c, "q50"], q.loc[c, "q85"]
        x_min, x_max = df[c].min(), df[c].max()
        mf_params[c] = {
            "LOW":  ("trap", (x_min, x_min, aL, bM)),
            "MED":  ("tri",  (aL, bM, dH)),
            "HIGH": ("trap", (bM, dH, x_max, x_max))
        }
    return mf_params, numeric_features

def mf_value(kind, params, x):
    return tri(x, *params) if kind == "tri" else trap(x, *params)

def fuzzify_row(values_dict, mf_params, numeric_features):
    """
    values_dict: {feature: float}
    """
    mem = {}
    for c in numeric_features:
        v = float(values_dict[c])
        sets = mf_params[c]
        mem[c] = {
            "LOW":  mf_value(sets["LOW"][0],  sets["LOW"][1],  v),
            "MED":  mf_value(sets["MED"][0],  sets["MED"][1],  v),
            "HIGH": mf_value(sets["HIGH"][0], sets["HIGH"][1], v),
        }
    return mem

# Konfiguracja kliniczna
RISK_UP_IF_HIGH = {
    "Body Mass Index (BMI)",
    "Hepatic Fat (%)",
    "Visceral Fat Area (cm^2)",
    "Triglyceride",
    "Fasting Blood Glucose",
    "Alanine Aminotransferase (ALT)",
    "Aspartate Aminotransferase (AST)",
    "Alkaline Phosphatase (ALP)",
    "C-Reactive Protein (CRP)",
    "Low Density Lipoprotein (LDL)",
    "Creatinine",
    "Age"
}
RISK_UP_IF_LOW = {"High Density Lipoprotein (HDL)", "Vitamin D"}

def infer_score(mem, risk_high_cols, risk_low_cols, weights=(0.6, 0.3, 0.1)):
    w2, w1, wprot = weights  # czytelnie: r2, r1, (1-prot)
    high_evidence = []
    for c in risk_high_cols:
        high_evidence.append(mem[c]["HIGH"])
    for c in risk_low_cols:
        high_evidence.append(mem[c]["LOW"])

    r1 = max(high_evidence) if high_evidence else 0.0
    topk = sorted(high_evidence, reverse=True)[:3]
    r2 = sum(topk) / max(len(topk), 1) if topk else 0.0

    protective = [mem[c]["HIGH"] for c in risk_low_cols] if risk_low_cols else []
    prot = max(protective) if protective else 0.0

    score = w2 * r2 + w1 * r1 + wprot * (1.0 - prot)
    return float(max(0.0, min(1.0, score)))

def evaluate_threshold(df, th):
    pred = (df["_score_"].values >= th).astype(int)
    y = df["_y_"].values
    tp = np.sum((pred == 1) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    youden = tpr - fpr
    acc = (tp + tn) / max(len(y), 1)
    return dict(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                tpr=float(tpr), fpr=float(fpr), youden=float(youden), acc=float(acc))

def sweep_roc(df):
    y = df["_y_"].values
    T = np.linspace(0, 1, 201)
    roc = []
    for th in T:
        pred = (df["_score_"].values >= th).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        roc.append((fpr, tpr))
    roc = np.array(roc)
    order = np.argsort(roc[:, 0])
    fpr_sorted, tpr_sorted = roc[order, 0], roc[order, 1]
    auc = safe_trapezoid(tpr_sorted, fpr_sorted)
    return fpr_sorted, tpr_sorted, auc

# ==============================
# 3) Pipeline Fuzzy dla GallStone
# ==============================
DEFAULT_TARGET = "Gallstone Status"
CANDIDATE_FEATURES = [
    "Age",
    "Body Mass Index (BMI)",
    "Fasting Blood Glucose",
    "Total Cholesterol",
    "High Density Lipoprotein (HDL)",
    "Low Density Lipoprotein (LDL)",
    "Triglyceride",
    "Aspartate Aminotransferase (AST)",
    "Alanine Aminotransferase (ALT)",
    "Alkaline Phosphatase (ALP)",
    "Creatinine",
    "Glomerular Filtration Rate (GFR)",
    "C-Reactive Protein (CRP)",
    "Hemoglobin (HGB)",
    "Vitamin D",
    "Hepatic Fat (%)",
    "Visceral Fat Area (cm^2)"
]

def run_fuzzy_fit(df, target_col=DEFAULT_TARGET, weights=(0.6, 0.3, 0.1), rnd=42):
    if target_col not in df.columns:
        raise ValueError(f"Nie znaleziono kolumny celu '{target_col}' w danych.")

    feature_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]
    work = df.copy()
    work["_y_"] = pd.to_numeric(work[target_col]).astype(int)

    mf_params, numeric_features = build_memberships(work, feature_cols)
    risk_high_cols = [c for c in numeric_features if c in RISK_UP_IF_HIGH]
    risk_low_cols  = [c for c in numeric_features if c in RISK_UP_IF_LOW]

    # score na całym zbiorze (do splitu)
    scores = []
    for _, r in work.iterrows():
        mem = fuzzify_row(r, mf_params, numeric_features)
        scores.append(infer_score(mem, risk_high_cols, risk_low_cols, weights))
    work["_score_"] = scores

    # split
    rng = np.random.RandomState(rnd)
    idx = np.arange(len(work))
    rng.shuffle(idx)
    split = int(0.7 * len(work))
    train_idx, test_idx = idx[:split], idx[split:]
    train, test = work.iloc[train_idx].copy(), work.iloc[test_idx].copy()

    # próg: Youden
    ths = np.linspace(0.2, 0.8, 121)
    best_th, best_youden = 0.5, -1e9
    for th in ths:
        m = evaluate_threshold(train, th)
        if m["youden"] > best_youden:
            best_youden, best_th = m["youden"], th

    # metryki test
    mtest = evaluate_threshold(test, best_th)
    fpr, tpr, auc = sweep_roc(test)

    # "model" to parametry MF, lista cech, próg i wagi
    model = {
        "numeric_features": numeric_features,
        "mf_params": mf_params,
        "risk_high_cols": risk_high_cols,
        "risk_low_cols": risk_low_cols,
        "threshold": float(best_th),
        "weights": tuple(float(w) for w in weights),
        "target_col": target_col
    }
    return model, work, train, test, mtest, (fpr, tpr, auc)

def predict_with_model(model, values_dict):
    numeric_features = model["numeric_features"]
    mf_params = model["mf_params"]
    risk_high_cols = model["risk_high_cols"]
    risk_low_cols  = model["risk_low_cols"]
    th = model["threshold"]
    weights = model.get("weights", (0.6, 0.3, 0.1))

    mem = fuzzify_row(values_dict, mf_params, numeric_features)
    score = infer_score(mem, risk_high_cols, risk_low_cols, weights)
    label = int(score >= th)

    # krótkie wyjaśnienie — wkłady
    parts = []
    for c in risk_high_cols:
        parts.append((c, "HIGH", mem[c]["HIGH"]))
    for c in risk_low_cols:
        parts.append((c, "LOW", mem[c]["LOW"]))
    parts.sort(key=lambda x: x[2], reverse=True)
    top = parts[:5]
    explanation = [f"{name} ({lvl}) → μ={val:.2f}" for name, lvl, val in top]

    return {"score": score, "label": label, "explanation": explanation, "membership": mem}

# ==============================
# 4) Wykresy (polskie etykiety)
# ==============================
def plot_score_hist(test_df, best_th):
    plt.figure()
    bins = np.linspace(0, 1, 21)
    s0 = test_df.loc[test_df["_y_"] == 0, "_score_"].values
    s1 = test_df.loc[test_df["_y_"] == 1, "_score_"].values
    plt.hist(s0, bins=bins, alpha=0.5, label="Brak kamicy (test)")
    plt.hist(s1, bins=bins, alpha=0.5, label="Kamica (test)")
    plt.axvline(best_th, linestyle="--")
    plt.title("Rozkład wyników Fuzzy – zbiór testowy")
    plt.legend()
    plt.show()

def plot_roc(fpr, tpr, auc):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Odsetek fałszywie dodatnich (FPR)")
    plt.ylabel("Odsetek prawdziwie dodatnich (TPR)")
    plt.title("Krzywa ROC – FuzzyLogic (test)")
    plt.legend()
    plt.show()

def plot_memberships(mf_params, df, wanted=("Body Mass Index (BMI)","Hepatic Fat (%)","Triglyceride","High Density Lipoprotein (HDL)","Age")):
    for c in wanted:
        if c not in mf_params or c not in df.columns:
            continue
        x_min, x_max = df[c].min(), df[c].max()
        x = np.linspace(x_min, x_max, 200)
        sets = mf_params[c]
        LOW  = np.array([mf_value(sets["LOW"][0],  sets["LOW"][1],  v) for v in x])
        MED  = np.array([mf_value(sets["MED"][0],  sets["MED"][1],  v) for v in x])
        HIGH = np.array([mf_value(sets["HIGH"][0], sets["HIGH"][1], v) for v in x])
        plt.figure()
        plt.plot(x, LOW, label="LOW")
        plt.plot(x, MED, label="MED")
        plt.plot(x, HIGH, label="HIGH")
        plt.title(f"Funkcje przynależności – {c}")
        plt.xlabel(c)
        plt.ylabel("Przynależność μ")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.show()

def plot_confusion_matrix(tp, fp, fn, tn):
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=int)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Macierz pomyłek (test)")
    plt.xticks([0, 1], ["Negatyw", "Pozytyw"])
    plt.yticks([0, 1], ["Negatyw", "Pozytyw"])
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.colorbar()
    plt.show()

def youden_univariate_threshold(df, feature, target=DEFAULT_TARGET, direction="high"):
        """
        Univariate: znajdź próg dla jednej cechy maksymalizujący Youdena.
        direction="high" -> pozytyw gdy x >= t
        direction="low"  -> pozytyw gdy x <= t
        Zwraca dict z: threshold, tpr, fpr, youden, pos_rate_above, support_above
        """
        if feature not in df.columns or target not in df.columns:
            return None
        x = pd.to_numeric(df[feature], errors="coerce").values
        y = pd.to_numeric(df[target], errors="coerce").astype(int).values

        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            return None

        # kandydaci progów: unikalne wartości x (ew. ograniczone do percentyli, żeby było stabilniej)
        xs = np.unique(x)
        if len(xs) > 400:  # przytnij do 400 równomiernych punktów (opcjonalnie)
            q = np.linspace(0.02, 0.98, 400)
            xs = np.unique(np.quantile(x, q))

        best = {"threshold": None, "tpr": 0.0, "fpr": 1.0, "youden": -1.0,
                "pos_rate_above": None, "support_above": None}

        for t in xs:
            if direction == "high":
                pred = (x >= t).astype(int)
                above = (x >= t)
            else:
                pred = (x <= t).astype(int)
                above = (x <= t)

            tp = np.sum((pred == 1) & (y == 1))
            tn = np.sum((pred == 0) & (y == 0))
            fp = np.sum((pred == 1) & (y == 0))
            fn = np.sum((pred == 0) & (y == 1))

            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            J = tpr - fpr

            if J > best["youden"]:
                best.update({
                    "threshold": float(t),
                    "tpr": float(tpr),
                    "fpr": float(fpr),
                    "youden": float(J),
                    "pos_rate_above": float(np.mean(y[above] == 1)) if np.any(above) else None,
                    "support_above": int(np.sum(above))
                })
        return best
# ==============================
# 5) Tkinter GUI
# ==============================




class FuzzyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GallStone – FuzzyLogic")
        self.geometry("840x520")

        self.file_path_var = tk.StringVar(value=os.path.abspath("dataset-uci.xlsx"))
        self.status_var = tk.StringVar(value="Wybierz plik i naucz model.")
        self.metrics_var = tk.StringVar(value="—")

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        # Wiersz wyboru pliku
        row1 = ttk.Frame(frm)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="Plik XLSX:").pack(side="left")
        ttk.Entry(row1, textvariable=self.file_path_var, width=60).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row1, text="Przeglądaj…", command=self.on_browse).pack(side="left")

        # Przyciski główne
        row2 = ttk.Frame(frm)
        row2.pack(fill="x", pady=8)
        ttk.Button(row2, text="Wczytaj dane", command=self.on_load).pack(side="left", padx=4)
        ttk.Button(row2, text="Naucz Fuzzy (fit)", command=self.on_fit).pack(side="left", padx=4)
        ttk.Button(row2, text="Wprowadź pacjenta (predict)", command=self.on_predict_dialog).pack(side="left", padx=4)
        ttk.Button(row2, text="Wykresy: Score", command=self.on_plot_score).pack(side="left", padx=4)
        ttk.Button(row2, text="Wykresy: ROC", command=self.on_plot_roc).pack(side="left", padx=4)
        ttk.Button(row2, text="Wykresy: MF", command=self.on_plot_mf).pack(side="left", padx=4)
        ttk.Button(row2, text="Wykresy: ConfMat", command=self.on_plot_cm).pack(side="left", padx=4)

        # Zapis/Wczytanie modelu
        row3 = ttk.Frame(frm)
        row3.pack(fill="x", pady=8)
        ttk.Button(row3, text="Zapisz model…", command=self.on_save_model).pack(side="left", padx=4)
        ttk.Button(row3, text="Wczytaj model…", command=self.on_load_model).pack(side="left", padx=4)

        # Status + metryki
        ttk.Label(frm, text="Status:").pack(anchor="w")
        ttk.Label(frm, textvariable=self.status_var, foreground="#333").pack(anchor="w", pady=(0,8))

        ttk.Label(frm, text="Metryki (test):").pack(anchor="w")
        self.metrics_box = tk.Text(frm, height=12)
        self.metrics_box.pack(fill="both", expand=True)

        # dane/pipeline
        self.df = None
        self.model = None
        self.split_cache = None  # (work, train, test, mtest, (fpr,tpr,auc))

    def on_browse(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files","*.xlsx"),("All files","*.*")])
        if path:
            self.file_path_var.set(path)

    def render_preview_table(self, df_preview, max_cols=40):
        """
        Renderuje DataFrame jako tabelkę ttk.Treeview z przewijaniem.
        Tworzy widżet przy pierwszym użyciu, potem tylko odświeża.
        """
        # Kontener na tabelę (tworzymy raz)
        if not hasattr(self, "preview_frame"):
            self.preview_frame = ttk.LabelFrame(self, text="Podgląd danych")
            self.preview_frame.pack(fill="both", expand=False, padx=12, pady=(6, 12))

            # Scrollbary
            self.prev_xscroll = ttk.Scrollbar(self.preview_frame, orient="horizontal")
            self.prev_yscroll = ttk.Scrollbar(self.preview_frame, orient="vertical")

            # Drzewo (tabelka)
            self.preview_tree = ttk.Treeview(
                self.preview_frame,
                columns=[],  # ustawimy przy odświeżeniu
                show="headings",
                xscrollcommand=self.prev_xscroll.set,
                yscrollcommand=self.prev_yscroll.set,
                height=8  # 8 wierszy
            )
            self.prev_xscroll.config(command=self.preview_tree.xview)
            self.prev_yscroll.config(command=self.preview_tree.yview)

            # Układ
            self.preview_tree.grid(row=0, column=0, sticky="nsew")
            self.prev_yscroll.grid(row=0, column=1, sticky="ns")
            self.prev_xscroll.grid(row=1, column=0, sticky="ew")

            self.preview_frame.columnconfigure(0, weight=1)
            self.preview_frame.rowconfigure(0, weight=1)

        # Ogranicz liczbę kolumn (żeby dało się ogarnąć szerokość)
        cols = list(df_preview.columns)[:max_cols]

        # Wyczyść stare kolumny i nagłówki
        self.preview_tree.delete(*self.preview_tree.get_children())
        for col in self.preview_tree["columns"]:
            self.preview_tree.heading(col, text="")
            self.preview_tree.column(col, width=0)

        # Ustaw nowe kolumny i nagłówki
        self.preview_tree["columns"] = cols
        for c in cols:
            self.preview_tree.heading(c, text=str(c))
            # ustal minimalną szerokość; można dopasować
            self.preview_tree.column(c, anchor="center", width=max(100, int(10 * len(str(c)))))

        # Wstaw dane wierszami (str() + nan-handling)
        for _, row in df_preview[cols].iterrows():
            values = [("" if pd.isna(v) else str(v)) for v in row.tolist()]
            self.preview_tree.insert("", "end", values=values)

    def on_load(self):
        path = self.file_path_var.get().strip()
        if not os.path.isfile(path):
            messagebox.showerror("Błąd", f"Nie znaleziono pliku:\n{path}")
            return
        try:
            self.df = try_read_excel(path)

            # Podstawowy status w pasku
            self.status_var.set(f"Wczytano: {os.path.basename(path)}  |  shape={self.df.shape}")

            # Przygotuj informacje o danych
            cols_all = list(self.df.columns)
            # cechy kandydujące zdefiniowane wcześniej
            feature_cols = [c for c in CANDIDATE_FEATURES if c in self.df.columns]
            numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(self.df[c])]

            # Podgląd pierwszych wierszy w czytelnym formacie (nie przycina kolumn)
            with pd.option_context('display.max_columns', None, 'display.width', 120):
                preview = self.df.head(8).to_string(index=False)

            # Info o kolumnie celu i rozkład klas (jeśli jest)
            target_info_lines = []
            if DEFAULT_TARGET in self.df.columns:
                try:
                    y = pd.to_numeric(self.df[DEFAULT_TARGET], errors="coerce").astype("Int64")
                    vc = y.value_counts(dropna=True).sort_index()
                    n0 = int(vc.get(0, 0))
                    n1 = int(vc.get(1, 0))
                    tot = n0 + n1
                    p0 = (n0 / tot * 100) if tot > 0 else 0.0
                    p1 = (n1 / tot * 100) if tot > 0 else 0.0
                    target_info_lines.append(f"Znaleziono kolumnę celu: '{DEFAULT_TARGET}'")
                    target_info_lines.append(f"Rozkład klas: 0={n0} ({p0:.1f}%) | 1={n1} ({p1:.1f}%) | razem={tot}")
                except Exception as _:
                    target_info_lines.append(
                        f"Znaleziono kolumnę celu: '{DEFAULT_TARGET}' (nie udało się policzyć rozkładu) ")
            else:
                target_info_lines.append(f"Uwaga: nie znaleziono kolumny celu '{DEFAULT_TARGET}'.")

            # Dtypes (skrót)
            dtypes_lines = [f"{c}: {str(self.df[c].dtype)}" for c in cols_all[:15]]
            if len(cols_all) > 15:
                dtypes_lines.append(f"... (+{len(cols_all) - 15} kolejnych kolumn)")

            # Złóż raport do okna tekstowego
            self.metrics_box.delete("1.0", "end")
            self.metrics_box.insert("end", f"Plik: {os.path.basename(path)}\n")
            self.metrics_box.insert("end", f"Wiersze×Kolumny: {self.df.shape[0]} × {self.df.shape[1]}\n\n")

            self.metrics_box.insert("end", "Kolumny (wszystkie):\n")
            self.metrics_box.insert("end", ", ".join(cols_all) + "\n\n")

            self.metrics_box.insert("end", "Typy danych (pierwsze 15 kolumn):\n")
            self.metrics_box.insert("end", "\n".join(dtypes_lines) + "\n\n")

            self.render_preview_table(self.df.head(self.df.shape[0]))

            self.metrics_box.insert("end", "\n".join(target_info_lines) + "\n\n")

            self.metrics_box.insert("end", f"Cechy kandydujące wykryte ({len(feature_cols)}):\n")
            self.metrics_box.insert("end", ", ".join(feature_cols) + "\n\n")

            self.metrics_box.insert("end", f"Cechy numeryczne używane przez Fuzzy ({len(numeric_features)}):\n")
            self.metrics_box.insert("end", ", ".join(numeric_features) + "\n")

        except Exception as e:
            messagebox.showerror("Błąd", f"Problem z plikiem: {e}")

    def on_fit(self):
        if self.df is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj dane.")
            return
        try:
            self.model, work, train, test, mtest, roc = run_fuzzy_fit(self.df, target_col=DEFAULT_TARGET)
            self.split_cache = (work, train, test, mtest, roc)
            th = self.model["threshold"]
            self.status_var.set(f"Nauczono model. Próg (Youden) = {th:.3f}")

            msg = []
            msg.append(f"Dokładność (accuracy) = {mtest['acc']:.3f} -> (TP+TN)/(TP+FP+FN+TN) ")
            msg.append(f"Czułość (TPR) = {mtest['tpr']:.3f} -> TP/(TP+FN)")
            msg.append(f"Odsetek fałszywie dodatnich (FPR) = {mtest['fpr']:.3f} -> FP/(FP+TN)")
            msg.append(f"Swoistość (TNR) = 1−FPR = {1 - mtest['fpr']:.3f} -> TN/(TN+FP")
            msg.append(f"Macierz: TP={mtest['tp']} FP={mtest['fp']} FN={mtest['fn']} "f"TN={mtest['tn']}")
            msg.append("")
            msg.append("legenda")
            msg.append("TP -> prawdziwie dodatnie")
            msg.append("FP -> fałszywie dodatnie")
            msg.append("FN -> fałszywie ujemne")
            msg.append("TN -> prawdziwie ujemne")

            # === Analiza progowa (pojedyncze cechy): gdzie „zaczyna się” ryzyko ===
            msg.append("Analiza progowa (pojedyncze cechy, próg Youdena univariate):")
            # Lista markerów do pokazania (jeśli są w danych)
            markers = [
                ("Body Mass Index (BMI)", "high", "pozytyw przy BMI ≥ t"),
                ("Fasting Blood Glucose", "high", "pozytyw przy Glucose ≥ t"),
                ("Triglyceride", "high", "pozytyw przy TG ≥ t"),
                ("High Density Lipoprotein (HDL)", "low", "pozytyw przy HDL ≤ t"),
            ]
            for feat, direction, note in markers:
                if feat in work.columns:
                    res = youden_univariate_threshold(work, feature=feat, target=DEFAULT_TARGET, direction=direction)
                    if res is not None and res["threshold"] is not None:
                        pr = res["pos_rate_above"]
                        supp = res["support_above"]
                        pr_txt = f"{pr * 100:.1f}%" if pr is not None else "n/d"
                        msg.append(
                            f"• {feat}: t ≈ {res['threshold']:.3f} ({note}); "
                            f"TPR={res['tpr']:.2f}, FPR={res['fpr']:.2f}, J={res['youden']:.2f}; "
                            f"częstość kamicy {'powyżej' if direction == 'high' else 'poniżej'} progu: {pr_txt} (n={supp})"
                        )
            msg.append("")

            self.metrics_box.delete("1.0", "end")
            self.metrics_box.insert("end", "\n".join(msg) + "\n")

            # zapis predykcji na zbiorze (do inspekcji)
            out = work.copy()
            out["Predicted"] = (out["_score_"] >= th).astype(int)
            save_path = os.path.join(os.path.dirname(self.file_path_var.get()), "gallstone_fuzzy_predictions.csv")
            cols = [self.model["target_col"]] + self.model["numeric_features"] + ["_score_", "Predicted"]
            cols = [c for c in cols if c in out.columns]
            out[cols].to_csv(save_path, index=False)
            self.metrics_box.insert("end", f"\nZapisano predykcje: {save_path}\n")

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się nauczyć Fuzzy: {e}")



    def on_predict_dialog(self):
        if self.model is None:
            messagebox.showwarning("Uwaga", "Najpierw naucz model (Naucz Fuzzy).")
            return

        # Tworzymy okno z polami dla numeric_features
        top = tk.Toplevel(self)
        top.title("Wprowadź pacjenta (predykcja)")
        frm = ttk.Frame(top, padding=10)
        frm.pack(fill="both", expand=True)

        feats = self.model["numeric_features"]
        entries = {}
        ttk.Label(frm, text="Wprowadź wartości cech (liczby):").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,8))

        # Podpowiedzi: mediany ze zbioru pracy (jeśli mamy cache)
        medians = {}
        if self.split_cache:
            work = self.split_cache[0]
            for c in feats:
                if c in work.columns:
                    try:
                        medians[c] = float(pd.to_numeric(work[c], errors="coerce").median())
                    except Exception:
                        medians[c] = 0.0

        for i, c in enumerate(feats, start=1):
            ttk.Label(frm, text=c).grid(row=i, column=0, sticky="w", padx=(0,8), pady=2)
            e = ttk.Entry(frm, width=20)
            e.grid(row=i, column=1, sticky="w", pady=2)
            if c in medians:
                e.insert(0, f"{medians[c]:.3f}")
            entries[c] = e

        out_box = tk.Text(frm, height=10, width=60)
        out_box.grid(row=len(feats)+1, column=0, columnspan=2, pady=(8,6))

        def do_predict():
            try:
                values = {}
                for c, w in entries.items():
                    values[c] = float(w.get().strip().replace(",", "."))
                res = predict_with_model(self.model, values)
                score = res["score"]
                label = res["label"]
                expl = res["explanation"]

                out_box.delete("1.0", "end")
                out_box.insert("end", f"Wynik (0–1): {score:.3f}\n")
                out_box.insert("end", f"Decyzja (etykieta): {label}\n")
                out_box.insert("end", "Największe wkłady (cecha – poziom – przynależność):\n")
                for line in expl:
                    out_box.insert("end", f"  • {line}\n")

            except Exception as ex:
                messagebox.showerror("Błąd", f"Nie udało się policzyć predykcji: {ex}")

        btn = ttk.Button(frm, text="Oblicz", command=do_predict)
        btn.grid(row=len(feats)+2, column=0, columnspan=2, pady=(4,0))

    def on_plot_score(self):
        if not self.split_cache:
            messagebox.showwarning("Uwaga", "Najpierw naucz model.")
            return
        work, train, test, mtest, roc = self.split_cache
        plot_score_hist(test, self.model["threshold"])

    def on_plot_roc(self):
        if not self.split_cache:
            messagebox.showwarning("Uwaga", "Najpierw naucz model.")
            return
        work, train, test, mtest, (fpr, tpr, auc) = self.split_cache
        plot_roc(fpr, tpr, auc)

    def on_plot_mf(self):
        if not self.split_cache:
            messagebox.showwarning("Uwaga", "Najpierw naucz model.")
            return
        work, train, test, mtest, roc = self.split_cache
        plot_memberships(self.model["mf_params"], work)

    def on_plot_cm(self):
        if not self.split_cache:
            messagebox.showwarning("Uwaga", "Najpierw naucz model.")
            return
        _, _, _, mtest, _ = self.split_cache
        plot_confusion_matrix(mtest["tp"], mtest["fp"], mtest["fn"], mtest["tn"])

    # --- Zapis/Wczytanie modelu ---
    def on_save_model(self):
        if self.model is None:
            messagebox.showwarning("Uwaga", "Brak modelu do zapisania. Najpierw 'Naucz Fuzzy'.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path:
            return
        # mf_params ma krotki i typy – zamieniamy na proste struktury
        to_save = {
            "numeric_features": self.model["numeric_features"],
            "risk_high_cols": self.model["risk_high_cols"],
            "risk_low_cols": self.model["risk_low_cols"],
            "threshold": self.model["threshold"],
            "weights": list(self.model.get("weights", (0.6, 0.3, 0.1))),
            "target_col": self.model["target_col"],
            "mf_params": {}
        }
        for feat, sets in self.model["mf_params"].items():
            to_save["mf_params"][feat] = {}
            for name, (kind, params) in sets.items():
                to_save["mf_params"][feat][name] = {"kind": kind, "params": list(params)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        self.status_var.set(f"Zapisano model do: {os.path.basename(path)}")

    def on_load_model(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json"),("All files","*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # odtwórz strukturę mf_params
            mf_params = {}
            for feat, sets in obj["mf_params"].items():
                mf_params[feat] = {}
                for name, item in sets.items():
                    mf_params[feat][name] = (item["kind"], tuple(item["params"]))
            self.model = {
                "numeric_features": obj["numeric_features"],
                "risk_high_cols": obj["risk_high_cols"],
                "risk_low_cols": obj["risk_low_cols"],
                "threshold": float(obj["threshold"]),
                "weights": tuple(obj.get("weights", [0.6, 0.3, 0.1])),
                "target_col": obj.get("target_col", DEFAULT_TARGET),
                "mf_params": mf_params
            }
            self.split_cache = None  # nie mamy train/test po wczytaniu z pliku
            self.status_var.set(f"Wczytano model z: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać modelu: {e}")

# ==============================
# 6) Main
# ==============================
if __name__ == "__main__":
    app = FuzzyApp()
    app.mainloop()
