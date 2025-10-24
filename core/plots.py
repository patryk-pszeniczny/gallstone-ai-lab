from matplotlib.figure import Figure
import numpy as np
from .metrics import roc_curve_vals, pr_curve_vals, safe_trapezoid, confusion

def plot_loss(history):
    fig = Figure(figsize=(7.5, 4.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(history.get("loss", []), label="loss (train)")
    if len(history.get("val_loss", [])) > 0:
        ax.plot(history.get("val_loss", []), label="loss (val)")
    ax.set_title("Przebieg uczenia (BCE loss)")
    ax.set_xlabel("Epoka"); ax.set_ylabel("Strata")
    ax.legend()
    return fig

def plot_roc(y_true, p):
    FPR, TPR = roc_curve_vals(y_true.reshape(-1), p.reshape(-1))
    auc = safe_trapezoid(TPR, FPR)
    fig = Figure(figsize=(7.5, 4.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(FPR, TPR, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC (test)")
    ax.legend()
    return fig

def plot_pr(y_true, p):
    REC, PREC = pr_curve_vals(y_true.reshape(-1), p.reshape(-1))
    aupr = safe_trapezoid(PREC, REC)
    fig = Figure(figsize=(7.5, 4.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(REC, PREC, label=f"AUPRC={aupr:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (test)")
    ax.legend()
    return fig

def plot_cm_metrics(y_true, p, thr: float):
    y_pred = (p.reshape(-1) >= thr).astype(int)
    tp, fp, fn, tn, tpr, fpr, prec, rec = confusion(y_true, y_pred)

    fig = Figure(figsize=(6.6, 4.5), dpi=100)
    ax = fig.add_subplot(121)
    cm = np.array([[tn, fp],[fn, tp]])
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"CM (thr={thr:.2f})")
    ax.set_xticks([0,1], ["Negatyw","Pozytyw"])
    ax.set_yticks([0,1], ["Negatyw","Pozytyw"])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(122)
    metrics = [("Precision", prec), ("Recall", rec), ("TPR", tpr), ("Specificity", 1-fpr)]
    ax2.bar([m[0] for m in metrics], [m[1] for m in metrics])
    ax2.set_ylim(0,1); ax2.set_title("Metryki")
    ax2.tick_params(axis='x', rotation=20)
    return fig
