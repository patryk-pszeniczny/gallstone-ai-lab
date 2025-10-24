import numpy as np
import pandas as pd
from .metrics import roc_curve_vals, safe_trapezoid

def make_test_table(Xte, yte, p_te, mu, sigma, features, test_idx=None):
    te_idx = test_idx if test_idx is not None else np.arange(len(yte))
    df_te = pd.DataFrame(Xte * sigma + mu, columns=features)
    df_te["y_true"] = yte.reshape(-1)
    df_te["p"] = p_te
    df_te["y_pred"] = (p_te >= 0.5).astype(int)
    df_te["row_id"] = te_idx
    df_te["error_type"] = np.where(
        (df_te["y_true"]==1)&(df_te["y_pred"]==0), "FN",
        np.where((df_te["y_true"]==0)&(df_te["y_pred"]==1), "FP","OK")
    )
    return df_te

def permutation_importance(model, Xte, yte, features, base_probs=None, seed=123):
    if base_probs is None:
        base_probs = model.predict_proba(Xte).reshape(-1)
    FPR, TPR = roc_curve_vals(yte.reshape(-1), base_probs)
    base_auc = safe_trapezoid(TPR, FPR)

    drops = []
    rng = np.random.RandomState(seed)
    for j, c in enumerate(features):
        Xperm = Xte.copy()
        rng.shuffle(Xperm[:, j])
        p_perm = model.predict_proba(Xperm).reshape(-1)
        FPRp, TPRp = roc_curve_vals(yte.reshape(-1), p_perm)
        aucp = safe_trapezoid(TPRp, FPRp)
        drops.append((c, base_auc - aucp))
    drops.sort(key=lambda x: x[1], reverse=True)
    return drops
