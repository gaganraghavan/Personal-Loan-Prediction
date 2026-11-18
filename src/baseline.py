"""Phase 3: paper-replication baselines (+ scaled control, + seed-variability study)."""
import numpy as np
import pandas as pd
from .models import baseline_estimators
from .evaluation import evaluate, fit_recording
from .preprocessing import load_data, split_data


def run_baselines(Xtr, ytr, Xte, yte, seed):
    rows, preds, params = [], {}, {}
    for version, scale_all in [("Baseline (paper-faithful)", False), ("Baseline (all scaled)", True)]:
        for name, model in baseline_estimators(seed, scale_all).items():
            secs, conv = fit_recording(model, Xtr, ytr)
            m, p = evaluate(model, Xte, yte)
            rows.append({"Model": name, "Version": version, **m,
                         "Fit seconds": round(secs, 2), "Convergence warning": conv})
            preds[(name, version)] = p
            params[f"{name} | {version}"] = {k: str(v) for k, v in model.get_params(deep=True).items()
                                             if "__" in k or k in ("memory", "verbose")}
    return pd.DataFrame(rows), preds, params


def seed_variability(seeds=range(10)):
    """Baseline sensitivity to the random split AND random initialisation.
    (The paper gives no seed, so single-run numbers can only be compared loosely.)"""
    X, y = load_data(); out = []
    for s in seeds:
        Xtr, Xte, ytr, yte = split_data(X, y, seed=s)
        for version, scale_all in [("Baseline (paper-faithful)", False)]:
            for name, model in baseline_estimators(s, scale_all).items():
                fit_recording(model, Xtr, ytr)
                m, _ = evaluate(model, Xte, yte)
                out.append({"Model": name, "Version": version, "Seed": s, **m})
    df = pd.DataFrame(out)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "Balanced Accuracy", "ROC-AUC", "PR-AUC", "FP", "FN"]
    agg = df.groupby(["Model", "Version"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{a} ({b})" for a, b in agg.columns]
    return df, agg.reset_index()
