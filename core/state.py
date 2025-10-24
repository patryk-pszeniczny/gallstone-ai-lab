import numpy as np
import pandas as pd

RANDOM_SEED = 42
TARGET_COL = "Gallstone Status"

class AppState:
    def __init__(self):
        # dane
        self.df: pd.DataFrame | None = None
        self.features: list[str] = []
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

        # split + z-score
        self.train_idx = None
        self.test_idx = None
        self.Xtr = None; self.ytr = None
        self.Xte = None; self.yte = None
        self.mu = None; self.sigma = None

        # model i wyniki
        self.model = None
        self.history: dict | None = None
        self.p_test = None
        self.last_eval = None  # (acc, tpr, fpr, tp, fp, fn, tn, auc, aupr)
        self.cal_platt = None  # (a,b)
        self.test_table: pd.DataFrame | None = None
