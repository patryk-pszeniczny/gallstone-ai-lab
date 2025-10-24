from .metrics import bce_loss
class Trainer:
    def __init__(self, model, Xtr, ytr, Xte=None, yte=None,
                 use_es=True, patience=20, use_step_lr=False, step_every=50, gamma=0.5,
                 on_epoch=None, on_done=None, stop_flag=None):
        import threading
        self.model = model
        self.Xtr, self.ytr = Xtr, ytr
        self.Xte, self.yte = Xte, yte
        self.use_es = use_es
        self.patience = patience
        self.use_step_lr = use_step_lr
        self.step_every = step_every
        self.gamma = gamma
        self.on_epoch = on_epoch or (lambda *args, **kwargs: None)
        self.on_done = on_done or (lambda *args, **kwargs: None)
        self.stop_flag = stop_flag or threading.Event()
        self.history = {"loss": [], "val_loss": []}

    def run(self):
        best_val = float("inf")
        best_epoch = 0
        best_weights = None
        epochs = self.model.epochs

        for ep in range(1, epochs + 1):
            if self.stop_flag.is_set():
                break
            if self.use_step_lr and ep>1 and (ep-1) % self.step_every == 0:
                self.model.lr *= self.gamma

            self.model.one_epoch(self.Xtr, self.ytr)
            y_hat_full = self.model.predict_proba(self.Xtr)
            loss = bce_loss(self.ytr, y_hat_full)
            val_loss = None
            if self.Xte is not None:
                y_val_hat = self.model.predict_proba(self.Xte)
                val_loss = bce_loss(self.yte, y_val_hat)

            self.history["loss"].append(loss)
            if val_loss is not None:
                self.history["val_loss"].append(val_loss)

            if self.use_es and val_loss is not None:
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_epoch = ep
                    best_weights = {
                        "W1": self.model.W1.copy(), "b1": self.model.b1.copy(),
                        "W2": self.model.W2.copy(), "b2": self.model.b2.copy()
                    }
                    if getattr(self.model, "h2", 0) > 0:
                        best_weights.update({"W3": self.model.W3.copy(), "b3": self.model.b3.copy()})
                elif ep - best_epoch >= self.patience:
                    break

            self.on_epoch(ep, loss, val_loss, self.model.lr)

        if best_weights is not None:
            self.model.W1 = best_weights["W1"]; self.model.b1 = best_weights["b1"]
            self.model.W2 = best_weights["W2"]; self.model.b2 = best_weights["b2"]
            if getattr(self.model, "h2", 0) > 0:
                self.model.W3 = best_weights["W3"]; self.model.b3 = best_weights["b3"]

        self.on_done(self.history)
