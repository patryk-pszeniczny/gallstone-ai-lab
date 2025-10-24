import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from core.preprocess import zscore_transform
from core.platt import apply_platt
from .base import BaseTab

class PredictTab(BaseTab):
    def build(self):
        top = ttk.Frame(self.tab); top.pack(fill="x")
        ttk.Label(top, text="Użyj po treningu. Wartości surowe; z-score stosuje się automatycznie.").pack(anchor="w")

        self.pred_frame = ttk.Frame(self.tab); self.pred_frame.pack(fill="both", expand=True, pady=(10,0))
        self.pred_entries = {}
        resf = ttk.LabelFrame(self.tab, text="Wynik"); resf.pack(fill="x", pady=(8,0))
        self.out = tk.Text(resf, height=8); self.out.pack(fill="both", expand=True)

        btns = ttk.Frame(self.tab); btns.pack(fill="x", pady=(6,0))
        ttk.Button(btns, text="Utwórz pola cech", command=self._build_fields).pack(side="left")
        ttk.Button(btns, text="Oblicz predykcję", command=self._predict).pack(side="left", padx=8)

    def _build_fields(self):
        for w in self.pred_frame.winfo_children():
            w.destroy()
        self.pred_entries.clear()
        if not self.state.features:
            self.warn("Uwaga","Najpierw wczytaj dane.")
            return
        cols = 5
        for i, c in enumerate(self.state.features):
            r, cc = i // cols, i % cols
            cell = ttk.Frame(self.pred_frame, padding=(0,2))
            cell.grid(row=r, column=cc, sticky="w", padx=8, pady=2)
            ttk.Label(cell, text=c).pack(anchor="w")
            e = ttk.Entry(cell, width=22); e.pack(anchor="w")
            self.pred_entries[c] = e

        if self.state.Xtr is not None:
            dftr = pd.DataFrame(self.state.Xtr * self.state.sigma + self.state.mu, columns=self.state.features)
            meds = dftr.median(numeric_only=True)
            for c,e in self.pred_entries.items():
                if c in meds.index:
                    try: e.insert(0, f"{float(meds[c]):.3f}")
                    except: pass

    def _predict(self):
        if self.state.model is None or self.state.mu is None:
            self.warn("Uwaga", "Najpierw wytrenuj model.")
            return
        try:
            x = np.zeros((1, len(self.state.features)), dtype=float)
            for j,c in enumerate(self.state.features):
                v = float(str(self.pred_entries[c].get()).replace(",", "."))
                x[0,j] = v
            xz = zscore_transform(x, self.state.mu, self.state.sigma)
            p = float(self.state.model.predict_proba(xz)[0,0])
            if self.state.cal_platt is not None:
                a,b = self.state.cal_platt
                p = float(apply_platt(np.array([p]), a, b)[0])
            yhat = int(p>=0.5)
            self.out.delete("1.0","end")
            self.out.insert("end", f"Prawdopodobieństwo (skalibrowane jeśli włączono): {p:.3f}\n")
            self.out.insert("end", f"Decyzja (próg 0.5): {yhat}\n")
        except Exception as e:
            self.error("Błąd", f"Nie udało się obliczyć predykcji: {e}")
