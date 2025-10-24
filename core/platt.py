import numpy as np

def fit_platt(y_true, y_score, max_iter=100, tol=1e-6):
    y = y_true.reshape(-1,1).astype(float)
    s = y_score.reshape(-1,1).astype(float)
    a = 0.0
    b = np.log(np.mean(y)+1e-6) - np.log(1-np.mean(y)+1e-6)
    for _ in range(max_iter):
        z = a*s + b
        p = 1/(1+np.exp(-z))
        g_a = np.sum((p - y)*s)
        g_b = np.sum(p - y)
        w = p*(1-p)
        H_aa = np.sum((s**2)*w)
        H_bb = np.sum(w)
        H_ab = np.sum(s*w)
        det = H_aa*H_bb - H_ab**2 + 1e-12
        da = ( H_bb*(-g_a) - (-H_ab)*(-g_b))/det
        db = ((-H_ab)*(-g_a) + H_aa*(-g_b))/det
        a_new = a + da; b_new = b + db
        if abs(da)+abs(db) < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    return float(a), float(b)

def apply_platt(y_score, a, b):
    z = a*y_score.reshape(-1) + b
    return 1/(1+np.exp(-z))
