import os, json
import numpy as np
from tkinter import ttk, filedialog
from .base import BaseTab
from core.mlp import MLPModel
from core.platt import fit_platt, apply_platt

class ModelTab(BaseTab):
    def build(self):
        row = ttk.Frame(self.tab); row.pack(fill="x")
        ttk.Button(row, text="Zapisz model…", command=self._save).pack(side="left")
        ttk.Button(row, text="Wczytaj model…", command=self._load).pack(side="left", padx=8)
        ttk.Button(row, text="Batch predykcja…", command=self._batch).pack(side="left", padx=8)

        cal = ttk.Frame(self.tab); cal.pack(fill="x", pady=(8,0))
        ttk.Button(cal, text="Naucz kalibrację (Platt) na TEŚCIE", command=self._learn_cal).pack(side="left")
        ttk.Button(cal, text="Wyczyść kalibrację", command=lambda: self._set_cal(None)).pack(side="left", padx=8)

        self.lbl_model = ttk.Label(self.tab, text="(brak modelu)"); self.lbl_model.pack(anchor="w", pady=(8,0))
        self.lbl_cal = ttk.Label(self.tab, text="Kalibracja: brak"); self.lbl_cal.pack(anchor="w")

    def _set_cal(self, ab):
        self.state.cal_platt = ab
        self.lbl_cal.config(text="Kalibracja: brak" if ab is None else f"Kalibracja (Platt): a={ab[0]:.3f}, b={ab[1]:.3f}")

    def _save(self):
        if self.state.model is None or self.state.mu is None:
            self.warn("Uwaga","Brak wytrenowanego modelu.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")], parent=self.root)
        if not path: return
        obj = {
            "features": self.state.features,
            "mu": self.state.mu.tolist(),
            "sigma": self.state.sigma.tolist(),
            "h1": self.state.model.h1,
            "h2": self.state.model.h2,
            "act": self.state.model.act,
            "l2": self.state.model.l2,
            "lr0": self.state.model.lr0,
            "weights": {}
        }
        obj["weights"]["W1"] = self.state.model.W1.tolist()
        obj["weights"]["b1"] = self.state.model.b1.tolist()
        if self.state.model.h2 > 0:
            obj["weights"]["W2"] = self.state.model.W2.tolist()
            obj["weights"]["b2"] = self.state.model.b2.tolist()
            obj["weights"]["W3"] = self.state.model.W3.tolist()
            obj["weights"]["b3"] = self.state.model.b3.tolist()
        else:
            obj["weights"]["W2"] = self.state.model.W2.tolist()
            obj["weights"]["b2"] = self.state.model.b2.tolist()
        if self.state.cal_platt is not None:
            obj["platt"] = {"a": self.state.cal_platt[0], "b": self.state.cal_platt[1]}
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        self.lbl_model.config(text=f"Zapisano: {os.path.basename(path)}")
        self.info("OK", f"Model zapisany do:\n{path}")

    def _load(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json"),("All files","*.*")], parent=self.root)
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.state.features = obj["features"]
            import numpy as np
            self.state.mu = np.array(obj["mu"])
            self.state.sigma = np.array(obj["sigma"])
            h1, h2 = int(obj["h1"]), int(obj["h2"])
            act = obj.get("act", "relu"); l2 = float(obj.get("l2",0.0)); lr0 = float(obj.get("lr0",1e-3))
            m = MLPModel(n_in=len(self.state.features), n_hidden=(h1,h2), activation=act, l2=l2)
            m.lr = lr0; m.lr0 = lr0
            W = obj["weights"]
            m.W1 = np.array(W["W1"]); m.b1 = np.array(W["b1"])
            if h2>0:
                m.W2 = np.array(W["W2"]); m.b2 = np.array(W["b2"])
                m.W3 = np.array(W["W3"]); m.b3 = np.array(W["b3"])
            else:
                m.W2 = np.array(W["W2"]); m.b2 = np.array(W["b2"])
            self.state.model = m

            pl = obj.get("platt")
            self._set_cal((float(pl["a"]), float(pl["b"]))) if pl else self._set_cal(None)
            self.lbl_model.config(text=f"Wczytano: {os.path.basename(path)}")
            self.info("OK", "Model wczytany. Przejdź do Predykcji.")
        except Exception as e:
            self.error("Błąd", f"Nie udało się wczytać modelu: {e}")

    def _learn_cal(self):
        if self.state.p_test is None or self.state.yte is None:
            self.warn("Uwaga","Najpierw wytrenuj model.")
            return
        try:
            a,b = fit_platt(self.state.yte.reshape(-1), self.state.p_test.reshape(-1))
            self._set_cal((a,b))
            self.info("Kalibracja", f"Nauczono Platt: a={a:.3f}, b={b:.3f}\nPredykcje będą wyświetlane po kalibracji.")
        except Exception as e:
            self.error("Błąd", f"Kalibracja się nie powiodła: {e}")

    def _batch(self):
        if self.state.model is None or self.state.mu is None:
            self.warn("Uwaga","Najpierw wytrenuj/wczytaj model.")
            return
        path = filedialog.askopenfilename(title="Wybierz plik do batch predykcji", filetypes=[("CSV/XLSX","*.csv;*.xlsx")], parent=self.root)
        if not path: return
        try:
            from core.dataset_io import read_dataset
            from core.preprocess import zscore_transform
            df = read_dataset(path)
            missing = [c for c in self.state.features if c not in df.columns]
            if missing:
                raise ValueError(f"Brak kolumn w pliku: {missing}")
            X = df[self.state.features].astype(float).values
            Xz = zscore_transform(X, self.state.mu, self.state.sigma)
            p = self.state.model.predict_proba(Xz).reshape(-1)
            if self.state.cal_platt is not None:
                a,b = self.state.cal_platt
                p = apply_platt(p, a, b)
            yhat = (p>=0.5).astype(int)
            out = df.copy()
            out["prob"] = p
            out["pred"] = yhat
            out_path = os.path.splitext(path)[0] + "_preds.csv"
            out.to_csv(out_path, index=False)
            self.info("Batch predykcja", f"Zapisano: {out_path}")
        except Exception as e:
            self.error("Błąd batch", f"{e}")
