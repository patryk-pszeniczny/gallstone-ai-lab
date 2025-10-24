import numpy as np

def safe_trapezoid(y, x):
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def bce_loss(y_true, y_prob, eps=1e-9):
    y_true = y_true.reshape(-1, 1)
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

def accuracy(y_true, y_pred):
    return np.mean((y_true.reshape(-1, 1) == y_pred.reshape(-1, 1)).astype(float))

def confusion(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tpr
    return tp, fp, fn, tn, tpr, fpr, prec, rec

def roc_curve_vals(y_true, y_score, num=200):
    thr = np.linspace(0, 1, num)
    TPR, FPR = [], []
    for t in thr:
        y_pred = (y_score >= t).astype(int)
        _, _, _, _, tpr, fpr, _, _ = confusion(y_true, y_pred)
        TPR.append(tpr); FPR.append(fpr)
    TPR = np.array(TPR); FPR = np.array(FPR)
    order = np.argsort(FPR)
    return FPR[order], TPR[order]

def pr_curve_vals(y_true, y_score, num=200):
    thr = np.linspace(0, 1, num)
    PREC, REC = [], []
    for t in thr:
        y_pred = (y_score >= t).astype(int)
        _, _, _, _, tpr, _, prec, rec = confusion(y_true, y_pred)
        PREC.append(prec); REC.append(rec)
    PREC = np.array(PREC); REC = np.array(REC)
    order = np.argsort(REC)
    return REC[order], PREC[order]

def binned_reliability(y_true, y_prob, bins=10):
    y_true = y_true.reshape(-1)
    y_prob = y_prob.reshape(-1)
    edges = np.linspace(0,1,bins+1)
    mids, acc, cnt = [], [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < bins-1 else (y_prob >= lo) & (y_prob <= hi)
        if np.any(mask):
            mids.append((lo+hi)/2)
            acc.append(np.mean(y_true[mask]==1))
            cnt.append(np.sum(mask))
    return np.array(mids), np.array(acc), np.array(cnt)
