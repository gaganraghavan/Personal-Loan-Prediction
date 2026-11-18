"""Metrics (positive class = loan accepted) and a fit helper that records warnings."""
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)


def get_scores(model, X):
    """Continuous score for the positive class (used for ROC / PR)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def compute_metrics(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": spec,
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, score),
        "PR-AUC": average_precision_score(y_true, score),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Predicted positives": int(tp + fp),
    }


def fit_recording(model, X, y):
    """Fit and report (fit_seconds, convergence_warning_raised)."""
    t = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.fit(X, y)
    conv = any(issubclass(i.category, ConvergenceWarning) for i in w)
    return time.time() - t, conv


def evaluate(model, X, y):
    pred = model.predict(X)
    score = get_scores(model, X)
    return compute_metrics(y, pred, score), {"y_true": np.asarray(y), "y_pred": pred, "score": score}
