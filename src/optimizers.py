"""Phase 6: controlled optimizer / solver comparisons.
Everything except the optimizer is held fixed. Selection is NOT done on the test set:
the CV columns (training data only) are the primary evidence; test numbers are reported
for transparency only."""
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from .evaluation import evaluate, fit_recording
from .preprocessing import make_pipeline

CV_SCORING = ["average_precision", "roc_auc", "f1", "recall"]


def _cv(model, Xtr, ytr, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        r = cross_validate(model, Xtr, ytr, cv=cv, scoring=CV_SCORING)
    return {f"CV {m}": r[f"test_{m}"].mean() for m in CV_SCORING} | {"CV PR-AUC std": r["test_average_precision"].std()}


def compare_mlp(Xtr, ytr, Xte, yte, seed, init_seeds=range(5)):
    configs = {
        "adam (lr=1e-3)": dict(solver="adam", learning_rate_init=1e-3),
        "sgd+momentum (lr=1e-3)": dict(solver="sgd", learning_rate_init=1e-3),
        "sgd+momentum (lr=1e-2)": dict(solver="sgd", learning_rate_init=1e-2),
        "lbfgs": dict(solver="lbfgs"),
    }
    rows = []
    for label, kw in configs.items():
        mk = lambda s: make_pipeline(MLPClassifier(hidden_layer_sizes=(100,), activation="relu", alpha=1e-4,
                                                   max_iter=500, random_state=s, **kw), scaled=True)
        cvm = _cv(mk(seed), Xtr, ytr, seed)
        tests = []
        for s in init_seeds:
            m = mk(s); secs, conv = fit_recording(m, Xtr, ytr)
            met, _ = evaluate(m, Xte, yte)
            tests.append({**met, "secs": secs, "conv": conv, "n_iter": m[-1].n_iter_})
        t = pd.DataFrame(tests)
        rows.append({"Model": "MLP", "Optimizer": label, **cvm,
                     **{f"Test {k} (mean)": t[k].mean() for k in ["Accuracy", "Recall", "F1", "ROC-AUC", "PR-AUC", "FP", "FN"]},
                     "Test PR-AUC (std over init seeds)": t["PR-AUC"].std(), "Test F1 (std over init seeds)": t["F1"].std(),
                     "Fit seconds (mean)": t["secs"].mean(), "Iterations (mean)": t["n_iter"].mean(),
                     "Runs hitting max_iter / ConvergenceWarning": int(t["conv"].sum())})
    return pd.DataFrame(rows)


def compare_logreg(Xtr, ytr, Xte, yte, seed):
    rows = []
    for solver in ["lbfgs", "newton-cg", "newton-cholesky", "liblinear", "sag", "saga"]:
        mk = lambda: make_pipeline(LogisticRegression(solver=solver, C=1.0, l1_ratio=0.0, max_iter=2000,
                                                      random_state=seed), scaled=True)
        cvm = _cv(mk(), Xtr, ytr, seed)
        m = mk(); secs, conv = fit_recording(m, Xtr, ytr); met, _ = evaluate(m, Xte, yte)
        rows.append({"Model": "Logistic Regression", "Optimizer": solver, **cvm,
                     **{f"Test {k}": met[k] for k in ["Accuracy", "Recall", "F1", "ROC-AUC", "PR-AUC", "FP", "FN"]},
                     "Fit seconds": secs, "Iterations": int(np.max(m[-1].n_iter_)), "ConvergenceWarning": conv})
    return pd.DataFrame(rows)


def compare_perceptron(Xtr, ytr, Xte, yte, seed):
    """Perceptron has no solver choice (it is SGD with the perceptron loss), so the only
    meaningful 'training configuration' axes are regularisation and the step size eta0."""
    rows = []
    for pen in [None, "l2", "l1", "elasticnet"]:
        for eta0 in [0.1, 1.0]:
            mk = lambda: make_pipeline(Perceptron(penalty=pen, alpha=1e-4, eta0=eta0, max_iter=1000, random_state=seed), scaled=True)
            cvm = _cv(mk(), Xtr, ytr, seed)
            m = mk(); fit_recording(m, Xtr, ytr); met, _ = evaluate(m, Xte, yte)
            rows.append({"Model": "Perceptron", "Optimizer": f"penalty={pen}, eta0={eta0}", **cvm,
                         **{f"Test {k}": met[k] for k in ["Accuracy", "Recall", "F1", "ROC-AUC", "PR-AUC", "FP", "FN"]}})
    return pd.DataFrame(rows)
