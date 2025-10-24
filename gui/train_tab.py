import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from core.state import RANDOM_SEED
from core.preprocess import stratified_split, zscore_fit, zscore_transform
from core.mlp import MLPModel
from core.metrics import accuracy, confusion, roc_curve_vals, pr_curve_vals, safe_trapezoid
from core.analysis import make_test_table
from core.trainer import Trainer
from .base import BaseTab

class TrainTab(BaseTab):
    def build(self):
        paned = ttk.Panedwindow(self.tab, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=8)
        right = ttk.Frame(paned, padding=8)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        # Ustawienia
        settings = ttk.LabelFrame(left, text="Ustawienia modelu", padding=8)
        settings.pack(fill="x")

        self.h1 = tk.IntVar(value=16)
        self.h2 = tk.IntVar(value=8)
        self.lr = tk.DoubleVar(value=1e-3)
        self.epochs = tk.IntVar(value=300)
        self.bs = tk.IntVar(value=32)
        self.l2 = tk.DoubleVar(value=1e-4)
        self.act = tk.StringVar(value="relu")
        self.split = tk.DoubleVar(value=0.7)

        self.use_cv = tk.BooleanVar(value=False)
        self.kfold = tk.IntVar(value=5)
        self.use_es = tk.BooleanVar(value=True)
        self.patience = tk.IntVar(value=20)
        self.use_step = tk.BooleanVar(value=False)
        self.step_every = tk.IntVar(value=50)
        self.gamma = tk.DoubleVar(value=0.5)

        def row(r, text, widget, helptext=None):
            ttk.Label(settings, text=text).grid(row=r, column=0, sticky="w", padx=(0,6), pady=3)
            widget.grid(row=r, column=1, sticky="we", pady=3)
            if helptext:
                ttk.Label(settings, text=helptext, foreground="#666").grid(row=r, column=2, sticky="w")
        settings.columnconfigure(1, weight=1)

        row(0, "Warstwa ukryta 1:", ttk.Spinbox(settings, from_=1, to=256, textvariable=self.h1, width=8), "neurony")
        row(1, "Warstwa ukryta 2:", ttk.Spinbox(settings, from_=0, to=256, textvariable=self.h2, width=8), "0 = brak")
        row(2, "Learning rate:", ttk.Entry(settings, textvariable=self.lr, width=10), "np. 0.001")
        row(3, "Epoki:", ttk.Spinbox(settings, from_=10, to=5000, increment=10, textvariable=self.epochs, width=8))
        row(4, "Batch size:", ttk.Spinbox(settings, from_=1, to=1024, textvariable=self.bs, width=8))
        row(5, "L2 (reg.):", ttk.Entry(settings, textvariable=self.l2, width=10), "np. 1e-4")
        row(6, "Aktywacja:", ttk.Combobox(settings, textvariable=self.act, values=["relu","tanh"], width=10))
        row(7, "Udział train:", ttk.Spinbox(settings, from_=0.5, to=0.9, increment=0.05, textvariable=self.split, width=6), "proporcja")

        extras = ttk.LabelFrame(left, text="CV / EarlyStopping / LR schedule", padding=8)
        extras.pack(fill="x", pady=(8,0))
        ttk.Checkbutton(extras, text="Użyj K-fold CV", variable=self.use_cv).grid(row=0, column=0, sticky="w")
        ttk.Label(extras, text="K:").grid(row=0, column=1, sticky="e")
        ttk.Spinbox(extras, from_=3, to=10, textvariable=self.kfold, width=5).grid(row=0, column=2, sticky="w", padx=4)
        ttk.Checkbutton(extras, text="Early stopping", variable=self.use_es).grid(row=1, column=0, sticky="w", pady=(4,0))
        ttk.Label(extras, text="patience:").grid(row=1, column=1, sticky="e")
        ttk.Spinbox(extras, from_=5, to=200, textvariable=self.patience, width=6).grid(row=1, column=2, sticky="w", padx=4)
        ttk.Checkbutton(extras, text="Step LR decay", variable=self.use_step).grid(row=2, column=0, sticky="w", pady=(4,0))
        ttk.Label(extras, text="co N epok:").grid(row=2, column=1, sticky="e")
        ttk.Spinbox(extras, from_=10, to=1000, textvariable=self.step_every, width=6).grid(row=2, column=2, sticky="w", padx=4)
        ttk.Label(extras, text="gamma:").grid(row=2, column=3, sticky="e")
        ttk.Entry(extras, textvariable=self.gamma, width=7).grid(row=2, column=4, sticky="w", padx=4)

        actions = ttk.Frame(left); actions.pack(fill="x", pady=(8,0))
        ttk.Button(actions, text="Podziel + Z-score", command=self._prep).pack(side="left")
        ttk.Button(actions, text="START trening", command=self._start).pack(side="left", padx=8)
        ttk.Button(actions, text="STOP", command=self._stop).pack(side="left")

        status = ttk.LabelFrame(right, text="Status", padding=8)
        status.pack(fill="x")
        self.prog = ttk.Progressbar(status, mode="determinate", maximum=100)
        self.prog.pack(fill="x")

        logf = ttk.LabelFrame(right, text="Logi treningu", padding=8)
        logf.pack(fill="both", expand=True, pady=(8,0))
        yscroll = ttk.Scrollbar(logf, orient="vertical")
        self.log = tk.Text(logf, height=20, wrap="word", yscrollcommand=yscroll.set)
        yscroll.config(command=self.log.yview)
        self.log.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        logf.columnconfigure(0, weight=1)
        logf.rowconfigure(0, weight=1)

        self._train_thread = None
        import threading
        self._stop_flag = threading.Event()

    def _println(self, s):
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.root.update_idletasks()

    def _prep(self):
        if self.state.df is None:
            self.warn("Uwaga", "Najpierw wczytaj dane w zakładce 'Dane'.")
            return
        try:
            train_idx, test_idx = stratified_split(self.state.X, self.state.y.reshape(-1), float(self.split.get()), RANDOM_SEED)
            self.state.train_idx, self.state.test_idx = train_idx, test_idx
            Xtr, ytr = self.state.X[train_idx], self.state.y[train_idx]
            Xte, yte = self.state.X[test_idx], self.state.y[test_idx]
            mu, sigma = zscore_fit(Xtr)
            self.state.mu, self.state.sigma = mu, sigma
            self.state.Xtr = zscore_transform(Xtr, mu, sigma); self.state.ytr = ytr
            self.state.Xte = zscore_transform(Xte, mu, sigma); self.state.yte = yte
            self._println(f"Split OK: train={len(self.state.Xtr)}, test={len(self.state.Xte)}, d={self.state.Xtr.shape[1]}")
            self._println("Z-score: policzono μ i σ na TRAIN i zastosowano do TRAIN/TEST.")
        except Exception as e:
            self.error("Błąd", f"Nie udało się przygotować danych: {e}")

    def _start(self):
        if self.state.X is None:
            self.warn("Uwaga", "Najpierw wczytaj dane.")
            return
        if self.use_cv.get():
            self._start_cv()
            return
        if self.state.Xtr is None:
            self.warn("Uwaga", "Najpierw kliknij 'Podziel + Z-score'.")
            return

        model = MLPModel(
            n_in=self.state.Xtr.shape[1],
            n_hidden=(int(self.h1.get()), int(self.h2.get()) if self.h2.get()>0 else 0),
            lr=float(self.lr.get()),
            epochs=int(self.epochs.get()),
            batch_size=int(self.bs.get()),
            activation=self.act.get(),
            l2=float(self.l2.get()),
            seed=RANDOM_SEED
        )
        self.state.model = model
        self.state.history = {"loss": [], "val_loss": []}
        self._stop_flag.clear()
        self.prog.configure(value=0, maximum=model.epochs)

        def on_epoch(ep, loss, val_loss, lr_now):
            self.state.history["loss"].append(loss)
            if val_loss is not None: self.state.history["val_loss"].append(val_loss)
            txt = f"Epoka {ep}/{model.epochs} | loss={loss:.4f}"
            if val_loss is not None: txt += f" | val_loss={val_loss:.4f}"
            txt += f" | lr={lr_now:.2e}"
            self._println(txt)
            self.prog.configure(value=ep)

        def on_done(history):
            self.state.history = history
            self._evaluate()

        trainer = Trainer(
            model, self.state.Xtr, self.state.ytr, self.state.Xte, self.state.yte,
            use_es=self.use_es.get(), patience=int(self.patience.get()),
            use_step_lr=self.use_step.get(), step_every=int(self.step_every.get()), gamma=float(self.gamma.get()),
            on_epoch=on_epoch, on_done=on_done, stop_flag=self._stop_flag
        )

        import threading
        def job():
            try:
                trainer.run()
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Błąd", f"Trening przerwany: {e}"))
        self._train_thread = threading.Thread(target=job, daemon=True)
        self._train_thread.start()
        self._println("Trening wystartował…")

    def _start_cv(self):
        # skrócony k-fold (jak wcześniej)
        k = int(self.kfold.get())
        h1, h2 = int(self.h1.get()), int(self.h2.get())
        lr, epochs, bs, l2, act = float(self.lr.get()), int(self.epochs.get()), int(self.bs.get()), float(self.l2.get()), self.act.get()

        def kfold_indices(y, k=5, seed=RANDOM_SEED):
            rng = np.random.RandomState(seed)
            idx0 = np.where(y.reshape(-1)==0)[0]
            idx1 = np.where(y.reshape(-1)==1)[0]
            rng.shuffle(idx0); rng.shuffle(idx1)
            folds0 = np.array_split(idx0, k)
            folds1 = np.array_split(idx1, k)
            folds = []
            for i in range(k):
                te = np.concatenate([folds0[i], folds1[i]])
                tr = np.setdiff1d(np.arange(len(y)), te, assume_unique=False)
                rng.shuffle(tr); rng.shuffle(te)
                folds.append((tr, te))
            return folds

        folds = kfold_indices(self.state.y, k)
        self.prog.configure(value=0, maximum=k)
        self._stop_flag.clear()

        import threading
        def job():
            aucs, accs = []
            aucs, accs = [], []
            try:
                for i, (tr, te) in enumerate(folds, start=1):
                    if self._stop_flag.is_set(): break
                    Xtr, ytr = self.state.X[tr], self.state.y[tr]
                    Xte, yte = self.state.X[te], self.state.y[te]
                    mu, sigma = zscore_fit(Xtr)
                    Xtrz, Xtez = zscore_transform(Xtr, mu, sigma), zscore_transform(Xte, mu, sigma)
                    from core.mlp import MLPModel
                    m = MLPModel(n_in=Xtrz.shape[1], n_hidden=(h1, h2 if h2>0 else 0),
                                 lr=lr, epochs=epochs, batch_size=bs, activation=act, l2=l2, seed=RANDOM_SEED)
                    for _ in range(epochs):
                        if self._stop_flag.is_set(): break
                        m.one_epoch(Xtrz, ytr)
                    p = m.predict_proba(Xtez).reshape(-1)
                    FPR, TPR = roc_curve_vals(yte.reshape(-1), p)
                    auc = safe_trapezoid(TPR, FPR)
                    acc = accuracy(yte, (p>=0.5).astype(int))
                    aucs.append(auc); accs.append(acc)
                    self.root.after(0, self._println, f"[CV] Fold {i}/{k}: AUC={auc:.3f}, Acc={acc:.3f}")
                    self.root.after(0, self.prog.configure, {"value": i})
                if len(aucs)>0:
                    self.root.after(0, self._println, f"[CV] Średnio: AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}, "
                                                      f"Acc={np.mean(accs):.3f}±{np.std(accs):.3f}")
                    self.root.after(0, messagebox.showinfo, "CV", "K-fold zakończony. Odznacz 'Użyj K-fold CV' i kliknij START.")
            except Exception as e:
                self.root.after(0, lambda: self.error("Błąd CV", str(e)))
        threading.Thread(target=job, daemon=True).start()
        self._println("K-fold CV wystartował…")

    def _stop(self):
        if self._train_thread and self._train_thread.is_alive():
            self._stop_flag.set()
            self._println("Żądanie STOP – czekam na zatrzymanie.")

    def _evaluate(self):
        if self.state.model is None or self.state.Xte is None:
            return
        p_tr = self.state.model.predict_proba(self.state.Xtr).reshape(-1)
        p_te = self.state.model.predict_proba(self.state.Xte).reshape(-1)

        yh_tr = (p_tr >= 0.5).astype(int)
        yh_te = (p_te >= 0.5).astype(int)

        acc_tr = accuracy(self.state.ytr, yh_tr)
        acc_te = accuracy(self.state.yte, yh_te)
        tp, fp, fn, tn, tpr, fpr, prec, rec = confusion(self.state.yte, yh_te)
        FPR, TPR = roc_curve_vals(self.state.yte.reshape(-1), p_te)
        auc = safe_trapezoid(TPR, FPR)
        REC, PREC = pr_curve_vals(self.state.yte.reshape(-1), p_te)
        aupr = safe_trapezoid(PREC, REC)

        self.state.p_test = p_te.reshape(-1,1)
        self.state.last_eval = (acc_te, tpr, fpr, tp, fp, fn, tn, auc, aupr)
        self.state.test_table = make_test_table(
            self.state.Xte, self.state.yte, p_te, self.state.mu, self.state.sigma, self.state.features, self.state.test_idx
        )

        self._println("")
        self._println("=== Wyniki (TEST) ===")
        self._println(f"Accuracy: {acc_te:.3f}")
        self._println(f"TPR (Recall): {tpr:.3f}")
        self._println(f"FPR: {fpr:.3f}")
        self._println(f"Specificity (1-FPR): {1-fpr:.3f}")
        self._println(f"Precision (próg 0.5): {prec:.3f}")
        self._println(f"Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
        self._println(f"AUC: {auc:.3f} | AUPRC: {aupr:.3f}")
