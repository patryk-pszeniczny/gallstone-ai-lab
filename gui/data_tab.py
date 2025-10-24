import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog
from core.state import TARGET_COL
from core.dataset_io import read_dataset, select_numeric_features
from .base import BaseTab

class DataTab(BaseTab):
    def build(self):
        top = ttk.Frame(self.tab)
        top.pack(fill="x")

        self.path_var = tk.StringVar(value=os.path.abspath("dataset-uci.xlsx"))
        ttk.Label(top, text="Plik danych (XLSX/CSV):").pack(side="left")
        ttk.Entry(top, textvariable=self.path_var, width=70).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(top, text="Przeglądaj…", command=self._browse).pack(side="left")
        ttk.Button(top, text="Wczytaj", command=self._load).pack(side="left", padx=(6, 0))

        self.lbl_info = ttk.Label(self.tab, text="(brak danych)")
        self.lbl_info.pack(anchor="w", pady=(8, 6))

        self._build_preview_table(self.tab)

    def _browse(self):
        path = filedialog.askopenfilename(filetypes=[("Excel/CSV","*.xlsx;*.csv"),("All files","*.*")], parent=self.root)
        if path:
            self.path_var.set(path)

    def _load(self):
        path = self.path_var.get().strip()
        if not os.path.isfile(path):
            self.error("Błąd", f"Nie znaleziono pliku:\n{path}")
            return
        try:
            df = read_dataset(path)
            if TARGET_COL not in df.columns:
                raise ValueError(f"Brak kolumny celu: '{TARGET_COL}'")
            feats = select_numeric_features(df, TARGET_COL)

            self.state.df = df
            self.state.features = feats

            y = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("Int64")
            vc = y.value_counts(dropna=True).sort_index()
            n0 = int(vc.get(0, 0)); n1 = int(vc.get(1, 0))
            self.lbl_info.config(text=f"Wczytano: {os.path.basename(path)} | shape={df.shape} | cechy={len(feats)} | 0={n0}, 1={n1}")

            self._render_preview(df)

            X = df[feats].astype(float).values
            y = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(int).values.reshape(-1, 1)
            self.state.X, self.state.y = X, y

            self.info("OK", "Dane wczytane. Przejdź do 'Trening'.")
        except Exception as e:
            self.error("Błąd", f"Problem z plikiem: {e}")

    def _build_preview_table(self, parent):
        frame = ttk.LabelFrame(parent, text="Podgląd danych")
        frame.pack(fill="both", expand=True)

        self.prev_xscroll = ttk.Scrollbar(frame, orient="horizontal")
        self.prev_yscroll = ttk.Scrollbar(frame, orient="vertical")

        self.tree = ttk.Treeview(
            frame, columns=[], show="headings",
            xscrollcommand=self.prev_xscroll.set, yscrollcommand=self.prev_yscroll.set, height=16
        )
        self.prev_xscroll.config(command=self.tree.xview)
        self.prev_yscroll.config(command=self.tree.yview)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.prev_yscroll.grid(row=0, column=1, sticky="ns")
        self.prev_xscroll.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    def _render_preview(self, df, max_rows=200, max_cols=40):
        dfp = df.head(max_rows).copy()
        cols = list(dfp.columns)[:max_cols]
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=str(c))
            self.tree.column(c, anchor="center", width=max(120, int(10 * len(str(c)))))
        for _, row in dfp[cols].iterrows():
            values = [("" if pd.isna(v) else str(v)) for v in row.tolist()]
            self.tree.insert("", "end", values=values)
