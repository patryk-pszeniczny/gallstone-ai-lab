import numpy as np
from .metrics import sigmoid

class MLPModel:
    def __init__(self, n_in, n_hidden=(16,8), lr=1e-3, epochs=200, batch_size=32, activation="relu", l2=0.0, seed=42):
        self.n_in = n_in
        self.h1 = n_hidden[0] if isinstance(n_hidden, (list, tuple)) and len(n_hidden) >= 1 else n_hidden
        self.h2 = n_hidden[1] if isinstance(n_hidden, (list, tuple)) and len(n_hidden) >= 2 else 0
        self.lr0 = lr
        self.lr = lr
        self.epochs = epochs
        self.bs = batch_size
        self.act = activation
        self.l2 = l2
        self.rng = np.random.RandomState(seed)
        self.W1 = self.rng.randn(self.n_in, self.h1) * np.sqrt(2 / self.n_in)
        self.b1 = np.zeros((1, self.h1))
        if self.h2 > 0:
            self.W2 = self.rng.randn(self.h1, self.h2) * np.sqrt(2 / self.h1)
            self.b2 = np.zeros((1, self.h2))
            self.W3 = self.rng.randn(self.h2, 1) * np.sqrt(1 / self.h2)
            self.b3 = np.zeros((1, 1))
        else:
            self.W2 = self.rng.randn(self.h1, 1) * np.sqrt(1 / self.h1)
            self.b2 = np.zeros((1, 1))

    def _act(self, z):
        if self.act == "tanh":
            return np.tanh(z)
        return np.maximum(0, z)

    def _act_grad(self, z):
        if self.act == "tanh":
            return 1 - np.tanh(z)**2
        return (z > 0).astype(float)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._act(z1)
        if self.h2 > 0:
            z2 = a1 @ self.W2 + self.b2
            a2 = self._act(z2)
            z3 = a2 @ self.W3 + self.b3
            y = sigmoid(z3)
            cache = (X, z1, a1, z2, a2, z3, y)
        else:
            z2 = a1 @ self.W2 + self.b2
            y = sigmoid(z2)
            cache = (X, z1, a1, z2, y)
        return y, cache

    def backward(self, cache, y_true):
        if self.h2 > 0:
            X, z1, a1, z2, a2, z3, y = cache
            N = X.shape[0]
            dy = (y - y_true) / N
            dW3 = a2.T @ dy + self.l2 * self.W3
            db3 = np.sum(dy, axis=0, keepdims=True)

            da2 = dy @ self.W3.T
            dz2 = da2 * self._act_grad(z2)
            dW2 = a1.T @ dz2 + self.l2 * self.W2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * self._act_grad(z1)
            dW1 = X.T @ dz1 + self.l2 * self.W1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            self.W3 -= self.lr * dW3; self.b3 -= self.lr * db3
            self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
        else:
            X, z1, a1, z2, y = cache
            N = X.shape[0]
            dy = (y - y_true) / N
            dW2 = a1.T @ dy + self.l2 * self.W2
            db2 = np.sum(dy, axis=0, keepdims=True)

            da1 = dy @ self.W2.T
            dz1 = da1 * self._act_grad(z1)
            dW1 = X.T @ dz1 + self.l2 * self.W1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

    def one_epoch(self, X, y):
        N = X.shape[0]
        idx = np.arange(N)
        self.rng.shuffle(idx)
        for s in range(0, N, self.bs):
            batch = idx[s:s+self.bs]
            Xb, yb = X[batch], y[batch]
            y_hat, cache = self.forward(Xb)
            self.backward(cache, yb)

    def predict_proba(self, X):
        y, _ = self.forward(X)
        return y

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)
