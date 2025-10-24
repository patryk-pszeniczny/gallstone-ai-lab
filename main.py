import tkinter as tk
from tkinter import ttk

from core.state import AppState, TARGET_COL
from gui.data_tab import DataTab
from gui.train_tab import TrainTab
from gui.predict_tab import PredictTab
from gui.plots_tab import PlotsTab
from gui.analysis_tab import AnalysisTab
from gui.model_tab import ModelTab
from gui.docs_tab import DocsTab


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GallStone – MLP (z-score + backprop, numpy)")
        self.geometry("1200x780")
        self.minsize(1024, 680)

        try:
            style = ttk.Style(self)
            style.theme_use("clam")
        except Exception:
            pass

        self.state = AppState()
        self._build_ui()

    def _build_ui(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_data = ttk.Frame(self.nb, padding=10)
        self.tab_train = ttk.Frame(self.nb, padding=10)
        self.tab_predict = ttk.Frame(self.nb, padding=10)
        self.tab_plots = ttk.Frame(self.nb, padding=10)
        self.tab_analysis = ttk.Frame(self.nb, padding=10)
        self.tab_model = ttk.Frame(self.nb, padding=10)
        self.tab_docs = ttk.Frame(self.nb, padding=10)

        self.nb.add(self.tab_data, text="Dane")
        self.nb.add(self.tab_train, text="Trening")
        self.nb.add(self.tab_predict, text="Predykcja")
        self.nb.add(self.tab_plots, text="Wykresy")
        self.nb.add(self.tab_analysis, text="Analiza (FP/FN, ważność)")
        self.nb.add(self.tab_model, text="Model")
        self.nb.add(self.tab_docs, text="Dokumentacja")

        # zakładki
        self.data_tab = DataTab(self, self.tab_data)
        self.train_tab = TrainTab(self, self.tab_train)
        self.predict_tab = PredictTab(self, self.tab_predict)
        self.plots_tab = PlotsTab(self, self.tab_plots)
        self.analysis_tab = AnalysisTab(self, self.tab_analysis)
        self.model_tab = ModelTab(self, self.tab_model)
        self.docs_tab = DocsTab(self, self.tab_docs)

        # budowa UI
        self.data_tab.build()
        self.train_tab.build()
        self.predict_tab.build()
        self.plots_tab.build()
        self.analysis_tab.build()
        self.model_tab.build()
        self.docs_tab.build()


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
