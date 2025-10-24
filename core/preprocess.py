import numpy as np

def stratified_split(X, y, train_frac=0.7, seed=42):
    rng = np.random.RandomState(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)
    n0 = int(len(idx0) * train_frac)
    n1 = int(len(idx1) * train_frac)
    train_idx = np.concatenate([idx0[:n0], idx1[:n1]])
    test_idx = np.concatenate([idx0[n0:], idx1[n1:]])
    rng.shuffle(train_idx); rng.shuffle(test_idx)
    return train_idx, test_idx

def zscore_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, ddof=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    return mu, sigma

def zscore_transform(X, mu, sigma):
    return (X - mu) / sigma
