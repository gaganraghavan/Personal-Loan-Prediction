"""Balanced SYNTHETIC-data experiment: predictive performance + computational efficiency of 6 models.

    python -m src.synthetic_experiment              # 5 seeds (42..46), full pipeline + report
    python -m src.synthetic_experiment --seeds 3    # fewer seeds
    python -m src.synthetic_experiment --quick      # 1 seed, small, smoke test -> results/synthetic_quick
    python -m src.synthetic_experiment --report-only

Per seed s: real 80/20 split (seed s) -> generator fitted on real TRAIN only -> 2,500 + 2,500 synthetic rows ->
stratified 80/20 split of the synthetic data -> per model: 5-fold StratifiedKFold tuning on synthetic TRAIN only
(refit objective: average precision, as in the main project) -> final fit -> evaluation on the synthetic TEST set
(first and only use) and, as an external validity check, on the untouched REAL test split (train-on-synthetic /
test-on-real). All models: MinMaxScaler inside the Pipeline (fitted on training data / CV training folds only).
Existing project files are not modified; this module reuses preprocessing, models, evaluation and tuning.
"""
import json, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from . import tuning
from .evaluation import evaluate
from .models import KMeansClassifier
from .preprocessing import load_data, make_pipeline, split_data
from .synthetic_data import TARGET, ClassConditionalCopula, quality_report

# Keep model warnings visible so convergence or fitting issues are not silently hidden.
ROOT = Path(__file__).resolve().parents[1]
MODELS = ["Perceptron", "Logistic Regression", "K-Means", "KNN", "SVM", "MLP"]
GRID_KEY = {"Perceptron": "Perceptron", "Logistic Regression": "Logistic Regression", "KNN": "KNeighbors", "SVM": "SVM"}
PERF = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "Balanced Accuracy", "ROC-AUC", "PR-AUC", "TP", "FP", "TN", "FN"]


def mlp_space():   # reduced vs the main project's MLP search to keep 5 seeds practical (documented)
    return {"clf__hidden_layer_sizes": [(50,), (100,), (64, 32)], "clf__activation": ["relu", "tanh"],
            "clf__solver": ["adam", "sgd", "lbfgs"], "clf__alpha": [1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [1e-3, 1e-2], "clf__batch_size": [64, 200], "clf__max_iter": [200]}


def tune(name, Xtr, ytr, seed, mlp_iter):
    """5-fold stratified CV on the training data only. Returns (unfitted best estimator, best params, seconds, #configs)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    if name == "MLP":
        pipe = make_pipeline(MLPClassifier(random_state=seed), True)
        gs = RandomizedSearchCV(pipe, mlp_space(), n_iter=mlp_iter, cv=cv, scoring="average_precision", n_jobs=1, random_state=seed, error_score=np.nan)
    else:
        _, est, space = tuning.search_spaces(seed)[GRID_KEY[name]]
        gs = GridSearchCV(make_pipeline(est, True), space, cv=cv, scoring="average_precision", n_jobs=1, error_score=np.nan)
    t = time.perf_counter(); gs.fit(Xtr, ytr); secs = time.perf_counter() - t   # includes the internal refit on the full training split
    return clone(gs.best_estimator_), {k.replace("clf__", ""): str(v) for k, v in gs.best_params_.items()}, secs, len(gs.cv_results_["params"])


def efficiency(model, Xte):
    """Prediction time on the whole test set (median of 5), average time per record, single-record latency (median of 100)."""
    batch = []
    for _ in range(5):
        t = time.perf_counter(); model.predict(Xte); batch.append(time.perf_counter() - t)
    single = []
    for i in range(100):
        t = time.perf_counter(); model.predict(Xte[i:i + 1]); single.append(time.perf_counter() - t)
    pt = float(np.median(batch))
    return {"Prediction time (s, test set)": pt, "Inference per record (us, batch avg)": pt / len(Xte) * 1e6,
            "Single-record latency (us, median)": float(np.median(single)) * 1e6}


def run_seed(seed, out, n_per_class, mlp_iter):
    X, y = load_data(); Xr_tr, Xr_te, yr_tr, yr_te = split_data(X, y, seed=seed)          # real split (same as the main project for seed 42)
    real_train, real_test = Xr_tr.assign(**{TARGET: yr_tr}), Xr_te.assign(**{TARGET: yr_te})
    gen = ClassConditionalCopula().fit(real_train); syn = gen.generate(n_per_class, seed)  # generator never sees the real test split
    syn.to_csv(out / "data" / f"synthetic_balanced_seed{seed}.csv", index=False)
    qres, qmarg, qeff = quality_report(real_train, real_test, syn, seed); qres["Generator acceptance rate (class 0 / 1)"] = str(gen.acceptance)
    Xs, ys = syn.drop(columns=TARGET), syn[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, random_state=seed, stratify=ys)
    Xtr, Xte, ytr, yte = Xtr.to_numpy(float), Xte.to_numpy(float), ytr.to_numpy(), yte.to_numpy()
    Xr, yr = Xr_te.to_numpy(float), yr_te.to_numpy()
    perf, eff, bests, preds = [], [], {}, {}
    for name in MODELS:
        if name == "K-Means":   # k=2 is fixed by the paper's design; nothing to tune with a supervised objective
            model, best, tune_s, ncfg = make_pipeline(KMeansClassifier(n_clusters=2, n_init=10, random_state=seed), True), {}, 0.0, 0
        else:
            model, best, tune_s, ncfg = tune(name, Xtr, ytr, seed, mlp_iter)
        ts = []                                                                              # final fit, timed separately from tuning: median of 3 fits
        for _ in range(3):
            m_ = clone(model); t = time.perf_counter(); m_.fit(Xtr, ytr); ts.append(time.perf_counter() - t)
        model, train_s = m_, float(np.median(ts))
        ms, ps = evaluate(model, Xte, yte); mr, _ = evaluate(model, Xr, yr)
        perf += [{"Seed": seed, "Model": name, "Eval set": "Synthetic test", **{k: ms[k] for k in PERF}},
                 {"Seed": seed, "Model": name, "Eval set": "Real test", **{k: mr[k] for k in PERF}}]
        eff.append({"Seed": seed, "Model": name, "Tuning time (s)": tune_s, "Configs evaluated": ncfg, "Training time (s)": train_s,
                    **efficiency(model, Xte), "Model size (KB)": len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024})
        bests[name] = best; preds[name] = ps
        print(f"  seed {seed} {name:20s} F1={ms['F1']:.3f} AUC={ms['ROC-AUC']:.3f} | tune {tune_s:6.1f}s train {train_s:5.2f}s", flush=True)
    return perf, eff, bests, preds, qres, qmarg, qeff


def main(argv):
    quick = "--quick" in argv; n_seeds = 1 if quick else (int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 5)
    out = ROOT / "results" / ("synthetic_quick" if quick else "synthetic")
    for d in ("data", "tables", "plots"): (out / d).mkdir(parents=True, exist_ok=True)
    n_per_class, mlp_iter = (800, 3) if quick else (2500, 8)
    if "--report-only" not in argv:
        P, E, B, Q = [], [], {}, []; t0 = time.time()
        for i in range(n_seeds):
            seed = 42 + i; print(f"seed {seed}", flush=True)
            perf, eff, bests, preds, qres, qmarg, qeff = run_seed(seed, out, n_per_class, mlp_iter)
            P += perf; E += eff; B[seed] = bests; Q.append(qres)
            if i == 0:
                pickle.dump(preds, open(out / "tables" / "seed42_synthetic_test_predictions.pkl", "wb"))
                qmarg.to_csv(out / "tables" / "quality_marginals_seed42.csv", index=False); qeff.to_csv(out / "tables" / "quality_effect_sizes_seed42.csv")
            pd.DataFrame(P).to_csv(out / "tables" / "per_seed_performance.csv", index=False); pd.DataFrame(E).to_csv(out / "tables" / "per_seed_efficiency.csv", index=False)
            pd.DataFrame(Q).to_csv(out / "tables" / "quality_per_seed.csv", index=False); json.dump(B, open(out / "tables" / "best_hyperparameters_per_seed.json", "w"), indent=1)
        json.dump({"seeds": [42 + i for i in range(n_seeds)], "rows_per_class": n_per_class, "mlp_random_candidates": mlp_iter,
                   "total_runtime_s": round(time.time() - t0, 1)}, open(out / "tables" / "run_config.json", "w"), indent=1)
    from .synthetic_report import build_report
    build_report(out)


if __name__ == "__main__":
    main(sys.argv[1:])
