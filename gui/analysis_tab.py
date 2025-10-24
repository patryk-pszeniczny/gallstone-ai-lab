import tkinter as tk
from tkinter import ttk
from core.analysis import permutation_importance
from matplotlib.figure import Figure
from .base import BaseTab

class AnalysisTab(BaseTab):
    def build(self):
        btns = ttk.Frame(self.tab); btns.pack(fill="x")
        ttk.Button(btns, text="FP/FN – odśwież", command=self._render_errors).pack(side="left", padx=6, pady=4)
        ttk.Button(btns, text="Ważność cech (permutacyjna)", command=self._importance).pack(side="left", padx=6, pady=4)
        ttk.Button(btns, text="Kalibracja (wykres)", command=self._reliability).pack(side="left", padx=6, pady=4)

        filt = ttk.Frame(self.tab); filt.pack(fill="x", pady=(4,0))
        ttk.Label(filt, text="Pokaż:").pack(side="left")
        self.filter_var = tk.StringVar(value="ALL")
        ttk.Combobox(filt, textvariable=self.filter_var, values=["ALL","FP","FN"], width=6).pack(side="left", padx=6)

        frame = ttk.LabelFrame(self.tab, text="Próbki (test)", padding=6)
        frame.pack(fill="both", expand=True, pady=(6,0))
        self.err_tree = ttk.Treeview(frame, columns=[], show="headings", height=16)
        ys = ttk.Scrollbar(frame, orient="vertical", command=self.err_tree.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=self.err_tree.xview)
        self.err_tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.err_tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1); frame.rowconfigure(0, weight=1)

        # plot area for analysis
        self.toolbar_area = ttk.Frame(self.tab); self.toolbar_area.pack(fill="x")
        self.plot_area = ttk.Frame(self.tab); self.plot_area.pack(fill="both", expand=True, pady=(4,0))

    def _render_errors(self):
        if self.state.test_table is None:
            self.warn("Uwaga","Najpierw wytrenuj model.")
            return
        df = self.state.test_table.copy()
        flt = self.filter_var.get()
        if flt in ("FP","FN"):
            df = df[df["error_type"]==flt]
        cols = ["row_id","y_true","y_pred","p","error_type"] + self.state.features[:min(10, len(self.state.features))]
        dfp = df[cols].copy()

        self.err_tree.delete(*self.err_tree.get_children())
        self.err_tree["columns"] = cols
        for c in cols:
            self.err_tree.heading(c, text=c)
            self.err_tree.column(c, anchor="center", width=max(90, int(9*len(c))))
        for _, row in dfp.iterrows():
            self.err_tree.insert("", "end", values=[("" if pd.isna(v) else str(v)) for v in row.tolist()])

    def _importance(self):
        if self.state.model is None or self.state.Xte is None:
            self.warn("Uwaga","Najpierw wytrenuj model.")
            return
        drops = permutation_importance(self.state.model, self.state.Xte, self.state.yte, self.state.features, self.state.p_test.reshape(-1))
        names = [d[0] for d in drops[:30]]
        vals = [d[1] for d in drops[:30]]

        fig = Figure(figsize=(7.5, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.barh(names[::-1], vals[::-1])
        ax.set_title("Permutation importance (spadek AUC po permutacji)")
        ax.set_xlabel("ΔAUC")
        self.draw_figure(fig, self.plot_area, self.toolbar_area)

    def _reliability(self):
        if self.state.p_test is None or self.state.yte is None:
            self.warn("Uwaga","Najpierw wytrenuj i oceń model.")
            return
        from core.metrics import binned_reliability
        mids, acc, cnt = binned_reliability(self.state.yte.reshape(-1), self.state.p_test.reshape(-1), bins=10)
        fig = Figure(figsize=(6.5, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot([0,1],[0,1], linestyle="--", label="idealnie")
        ax.plot(mids, acc, marker="o", label="empirycznie")
        ax.set_xlabel("Przewidziane P(1)")
        ax.set_ylabel("Częstość rzeczywista")
        ax.set_title("Reliability diagram (test)")
        ax.legend()
        self.draw_figure(fig, self.plot_area, self.toolbar_area)
