import tkinter as tk
from tkinter import ttk
from core.plots import plot_loss, plot_roc, plot_pr, plot_cm_metrics
from .base import BaseTab

class PlotsTab(BaseTab):
    def build(self):
        top = ttk.Frame(self.tab); top.pack(fill="x")
        ttk.Button(top, text="Loss", command=self._loss).pack(side="left", padx=6, pady=4)
        ttk.Button(top, text="ROC", command=self._roc).pack(side="left", padx=6, pady=4)
        ttk.Button(top, text="Precision-Recall", command=self._pr).pack(side="left", padx=6, pady=4)
        ttk.Button(top, text="CM (z progiem)", command=self._cm).pack(side="left", padx=6, pady=4)

        thr_box = ttk.Frame(self.tab); thr_box.pack(fill="x")
        ttk.Label(thr_box, text="Próg:").pack(side="left")
        self.thr_var = tk.DoubleVar(value=0.5)
        thr = ttk.Scale(thr_box, from_=0.0, to=1.0, variable=self.thr_var, command=lambda e: None)
        thr.pack(side="left", fill="x", expand=True, padx=8)
        self.thr_label = ttk.Label(thr_box, text="0.50"); self.thr_label.pack(side="left")
        def _upd(*_): self.thr_label.config(text=f"{self.thr_var.get():.2f}")
        self.thr_var.trace_add("write", _upd)

        self.toolbar_area = ttk.Frame(self.tab); self.toolbar_area.pack(fill="x")
        self.plot_area = ttk.Frame(self.tab); self.plot_area.pack(fill="both", expand=True, pady=(4,0))

    def _loss(self):
        if not self.state.history or len(self.state.history.get("loss",[]))==0:
            self.warn("Uwaga","Brak historii treningu.")
            return
        fig = plot_loss(self.state.history)
        self.draw_figure(fig, self.plot_area, self.toolbar_area)

    def _roc(self):
        if self.state.p_test is None or self.state.yte is None:
            self.warn("Uwaga","Najpierw wytrenuj i oceń model.")
            return
        fig = plot_roc(self.state.yte, self.state.p_test)
        self.draw_figure(fig, self.plot_area, self.toolbar_area)

    def _pr(self):
        if self.state.p_test is None or self.state.yte is None:
            self.warn("Uwaga","Najpierw wytrenuj i oceń model.")
            return
        fig = plot_pr(self.state.yte, self.state.p_test)
        self.draw_figure(fig, self.plot_area, self.toolbar_area)

    def _cm(self):
        if self.state.p_test is None or self.state.yte is None:
            self.warn("Uwaga","Najpierw wytrenuj i oceń model.")
            return
        fig = plot_cm_metrics(self.state.yte, self.state.p_test, float(self.thr_var.get()))
        self.draw_figure(fig, self.plot_area, self.toolbar_area)
