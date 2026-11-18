"""Methodological verification pass. ADDITIVE: reads saved results / re-runs pieces, never alters
the experiment code or the saved experiment outputs.   python -m src.verification {quick|lr|km|opt}"""
import inspect, json, pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from . import tuning
from .evaluation import evaluate, fit_recording
from .models import baseline_estimators
from .preprocessing import DATA_PATH, SEED, TARGET, load_data, make_pipeline, split_data

RES = Path(__file__).resolve().parents[1] / "results"; OUT = RES / "verification"; OUT.mkdir(exist_ok=True)
SRC = Path(__file__).resolve().parent
warnings.filterwarnings("ignore")


def _split():
    X, y = load_data(); return X, y, *split_data(X, y, seed=SEED)


# ---------- items 1 & 3: baseline parameters, scaler fitting ----------
def check_params_and_scaling():
    X, y, Xtr, Xte, ytr, yte = _split(); rows = []
    for name, pipe in baseline_estimators(SEED, False).items():
        est = pipe[-1]; d = (KMeans() if name == "K-Means" else est.__class__()).get_params()
        diff = {k: [str(d.get(k)), str(v)] for k, v in est.get_params().items() if str(d.get(k)) != str(v)}
        rows.append({"Model": name, "Scaler in pipeline": "scaler" in pipe.named_steps,
                     "Params differing from sklearn defaults [default, used]": json.dumps(diff)})
    pd.DataFrame(rows).to_csv(OUT / "1_baseline_param_audit.csv", index=False)
    pipe = baseline_estimators(SEED, False)["MLP"]; fit_recording(pipe, Xtr, ytr); sc = pipe[0]
    differ = [c for c in X.columns if X[c].max() != Xtr[c].max() or X[c].min() != Xtr[c].min()]
    idx = [list(X.columns).index(c) for c in differ]
    res = {"scaler.data_max_/min_ equal TRAIN max/min": bool(np.allclose(sc.data_max_, Xtr.max().values) and np.allclose(sc.data_min_, Xtr.min().values)),
           "columns where full-data min/max differ from train min/max (leak would be detectable)": differ,
           "scaler.data_max_ equals FULL-data max on those columns (would indicate leak)": bool(np.allclose(sc.data_max_[idx], X[differ].max().values)) if differ else None}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    mk = lambda: make_pipeline(LogisticRegression(max_iter=2000, random_state=SEED), True)
    auto = cross_validate(mk(), Xtr, ytr, cv=cv, scoring="average_precision")["test_score"]
    man, leak = [], []
    gsc = MinMaxScaler().fit(Xtr)
    for tr, va in cv.split(Xtr, ytr):
        s = MinMaxScaler().fit(Xtr.iloc[tr]); m = LogisticRegression(max_iter=2000).fit(s.transform(Xtr.iloc[tr]), ytr.iloc[tr])
        man.append(average_precision_score(ytr.iloc[va], m.predict_proba(s.transform(Xtr.iloc[va]))[:, 1]))
        m2 = LogisticRegression(max_iter=2000).fit(gsc.transform(Xtr.iloc[tr]), ytr.iloc[tr])
        leak.append(average_precision_score(ytr.iloc[va], m2.predict_proba(gsc.transform(Xtr.iloc[va]))[:, 1]))
    res["CV: Pipeline score vs manual per-fold-scaler score, max abs diff"] = float(np.max(np.abs(auto - np.array(man))))
    res["CV: (illustration) leaky scaler fitted on all training rows vs correct, max abs diff"] = float(np.max(np.abs(np.array(leak) - np.array(man))))
    src = {f.name: [i + 1 for i, l in enumerate(f.read_text().splitlines()) if "MinMaxScaler(" in l or "StandardScaler(" in l] for f in SRC.glob("*.py")}
    res["lines instantiating a scaler, per source file (excluding verification.py)"] = {k: v for k, v in src.items() if v and k != "verification.py"}
    json.dump(res, open(OUT / "3_scaling_audit.json", "w"), indent=2); print(json.dumps(res, indent=1))


# ---------- item 2: test set never used for selection ----------
def check_no_test_in_selection():
    X, y, Xtr, Xte, ytr, yte = _split(); saved = json.load(open(RES / "best_hyperparameters.json"))
    tuned = pd.read_csv(RES / "tuned_results.csv").set_index("Model"); rows = []
    for name, (kind, est, space) in tuning.search_spaces(SEED).items():
        if name == "MLP": continue   # re-running costs ~5.5 min; code path identical
        gs, _, _ = tuning.run_search(name, kind, est, space, Xtr, ytr, SEED)
        got = {k.replace("clf__", ""): str(v) for k, v in gs.best_params_.items()}
        rows.append({"Model": name, "rerun best params == saved": got == saved[name],
                     "rerun best CV PR-AUC": gs.best_score_, "saved best CV PR-AUC": tuned.loc[name, "Best CV PR-AUC"]})
    pd.DataFrame(rows).to_csv(OUT / "2_selection_rerun.csv", index=False); print(pd.DataFrame(rows).to_string())
    lines = []
    for f in sorted(SRC.glob("*.py")):
        for i, l in enumerate(f.read_text().splitlines()):
            if "Xte" in l or "yte" in l: lines.append(f"{f.name}:{i+1}: {l.strip()}")
    open(OUT / "2_test_set_usage_lines.txt", "w").write("run_search signature: %s\n\n" % list(inspect.signature(tuning.run_search).parameters) + "\n".join(lines))
    print("tuning.py mentions of Xte/yte:", [l for l in lines if l.startswith("tuning.py")])


# ---------- item 8: metrics vs saved predictions ----------
def manual_metrics(y, pred, score):
    y = np.asarray(y); pred = np.asarray(pred); score = np.asarray(score, float)
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn); spec = tn / (tn + fp)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    n1 = y.sum(); n0 = len(y) - n1; r = rankdata(score); auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    o = np.argsort(-score, kind="mergesort"); s = score[o]; yy = y[o]
    idx = np.r_[np.where(np.diff(s))[0], len(s) - 1]; tps = np.cumsum(yy)[idx]; fps = np.cumsum(1 - yy)[idx]
    ap = float(np.sum(np.diff(np.r_[0, tps / n1]) * (tps / (tps + fps))))
    return {"Accuracy": (tp + tn) / len(y), "Precision": prec, "Recall": rec, "F1": f1, "Specificity": spec,
            "Balanced Accuracy": (rec + spec) / 2, "ROC-AUC": auc, "PR-AUC": ap, "FP": fp, "FN": fn, "TP": tp, "TN": tn}


def check_metrics():
    X, y, Xtr, Xte, ytr, yte = _split(); preds = pickle.load(open(RES / "predictions.pkl", "rb"))
    comp = pd.read_csv(RES / "model_comparison.csv"); rows = []
    thr = {"Perceptron": 0.0, "SVM": 0.0, "K-Means": 0.0, "Logistic Regression": 0.5, "KNeighbors": 0.5, "MLP": 0.5}
    for _, r in comp.iterrows():
        p = preds[(r["Model"], r["Version"])]; m = manual_metrics(p["y_true"], p["y_pred"], p["score"])
        d = max(abs(m[k] - r[k]) for k in m)
        rows.append({"Model": r["Model"], "Version": r["Version"], "max abs diff vs model_comparison.csv": d,
                     "y_true == test labels (seed 42 split)": bool((np.asarray(p["y_true"]) == yte.values).all()),
                     "y_pred == (score > threshold)": bool(((np.asarray(p["score"]) > thr[r["Model"]]).astype(int) == p["y_pred"]).all())})
    v = pd.DataFrame(rows); v.to_csv(OUT / "8_metric_recomputation.csv", index=False)
    b = pd.read_csv(RES / "baseline_results.csv"); t = pd.read_csv(RES / "tuned_results.csv")
    src = pd.concat([b, t]).merge(comp, on=["Model", "Version"], suffixes=("", "_c"))
    cols = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "Balanced Accuracy", "ROC-AUC", "PR-AUC", "FP", "FN"]
    print(v.to_string()); print("rows:", len(v), "| max diff overall:", v.iloc[:, 2].max(),
          "| baseline/tuned csv vs comparison csv max diff:", max(abs(src[c] - src[c + "_c"]).max() for c in cols))


# ---------- item 10: MLP search size ----------
def check_mlp_search():
    kind, est, space = tuning.search_spaces(SEED)["MLP"]; n = int(np.prod([len(v) for v in space.values()]))
    tuned = pd.read_csv(RES / "tuned_results.csv").set_index("Model")
    res = {"MLP search space size (all combinations)": n, "candidates evaluated": int(tuned.loc["MLP", "Configs evaluated"]),
           "fraction of space sampled": tuned.loc["MLP", "Configs evaluated"] / n, "search type": tuned.loc["MLP", "Search type"],
           "grid sizes of other models": {k: int(v["Configs evaluated"]) for k, v in tuned.iterrows() if k != "MLP"}}
    docs = {f: [l for l in open(Path(__file__).resolve().parents[1] / f).read().splitlines() if "30" in l and ("MLP" in l or "candidate" in l)] for f in ["README.md", "SUMMARY.md"]}
    res["doc lines mentioning the 30-candidate search"] = docs
    json.dump(res, open(OUT / "10_mlp_search.json", "w"), indent=2); print(json.dumps(res, indent=1))


# ---------- item 6: Logistic Regression discrepancy ----------
def lr_study(n_seeds=20):
    df = pd.read_csv(DATA_PATH); y = df[TARGET].astype(int); base = df.drop(columns=[TARGET, "ID"])
    Xs = {"base": base, "drop ZIP": base.drop(columns=["ZIP Code"]), "ID": df.drop(columns=[TARGET]),
          "edu": pd.get_dummies(base, columns=["Education"], dtype=int)}
    V = {"default (paper-faithful: unscaled, lbfgs, C=1, max_iter=100)": ("base", None, {}),
         "max_iter=1000": ("base", None, {"max_iter": 1000}), "max_iter=10000": ("base", None, {"max_iter": 10000}),
         "drop ZIP Code": ("drop ZIP", None, {}), "include ID column": ("ID", None, {}),
         "Education one-hot": ("edu", None, {}), "MinMax scaled": ("base", MinMaxScaler, {}),
         "Standard scaled": ("base", StandardScaler, {}), "solver=liblinear": ("base", None, {"solver": "liblinear"}),
         "C=1e6 (~unregularised), max_iter=10000": ("base", None, {"C": 1e6, "max_iter": 10000})}
    rows = []
    for strat in (True, False):
        for seed in range(n_seeds):
            for name, (xk, sc, kw) in V.items():
                Xtr, Xte, ytr, yte = train_test_split(Xs[xk], y, test_size=.2, random_state=seed, stratify=y if strat else None)
                steps = ([sc()] if sc else []) + [LogisticRegression(**{"max_iter": 100, **kw})]
                pipe = Pipeline([(f"s{i}", s) for i, s in enumerate(steps)])
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always"); pipe.fit(Xtr, ytr)
                conv = any(issubclass(i.category, ConvergenceWarning) for i in w)
                pr = pipe.predict_proba(Xte)[:, 1]; pred = pipe.predict(Xte)
                rows.append({"Variant": name, "Stratified": strat, "Seed": seed, "ROC-AUC": roc_auc_score(yte, pr),
                             "ROC-AUC (hard labels)": roc_auc_score(yte, pred), "Predicted positives": int(pred.sum()),
                             "Recall": float(pred[yte.values == 1].mean()), "ConvergenceWarning": conv})
        print("strat", strat, "done", flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT / "6_logreg_variant_runs.csv", index=False)
    g = d.groupby(["Variant", "Stratified"]); s = g.agg(**{"AUC mean": ("ROC-AUC", "mean"), "AUC min": ("ROC-AUC", "min"), "AUC max": ("ROC-AUC", "max"),
        "mean predicted positives": ("Predicted positives", "mean"), "frac runs with 0 predicted positives": ("Predicted positives", lambda v: (v == 0).mean()),
        "frac runs AUC in [0.78,0.84]": ("ROC-AUC", lambda v: ((v >= .78) & (v <= .84)).mean()),
        "frac runs ConvergenceWarning": ("ConvergenceWarning", "mean")}).reset_index()
    s.to_csv(OUT / "6_logreg_variant_summary.csv", index=False); print(s.round(3).to_string())


# ---------- item 7: K-Means discrepancy ----------
def implied_auc(fpr, tpr): return 0.5 * fpr * tpr + (1 - fpr) * (tpr + 1) / 2


def km_run(Xtr, ytr, Xte, yte, scale, init, n_init, mapping, seed):
    if scale: s = MinMaxScaler().fit(Xtr); A, B = s.transform(Xtr), s.transform(Xte)
    else: A, B = Xtr.values, Xte.values
    km = KMeans(2, init=init, n_init=n_init, random_state=seed).fit(A)
    if mapping == "train positive rate":
        rates = [ytr.values[km.labels_ == c].mean() if (km.labels_ == c).any() else 0 for c in (0, 1)]; pos = int(np.argmax(rates))
    else: pos = 1
    d = km.transform(B); score = d[:, 1 - pos] - d[:, pos]; pred = (score > 0).astype(int); yt = yte.values
    return {"AUC (score)": roc_auc_score(yt, score), "AUC (hard labels)": roc_auc_score(yt, pred), "TPR": pred[yt == 1].mean(),
            "FPR": pred[yt == 0].mean(), "FP": int(((pred == 1) & (yt == 0)).sum()), "FN": int(((pred == 0) & (yt == 1)).sum())}


def km_study(n_seeds=20):
    X, y = load_data(); out = {}
    out["implied AUC if text numbers (7 FP / 72 FN) held on a 1000-row test set, by #test positives 85..110"] = {
        int(n): round(implied_auc(7 / (1000 - n), (n - 72) / n), 3) for n in range(85, 111, 5)}
    out["implied AUC of a single-vertex curve at the figure's KNN vertex (FPR~0.10, TPR~0.59; read by eye, +-0.02)"] = {
        "central": round(implied_auc(.10, .59), 3), "range": [round(implied_auc(.12, .57), 3), round(implied_auc(.08, .61), 3)]}
    out["unscaled feature std (what K-Means distance sees)"] = X.std().round(1).to_dict()
    Xtr, Xte, ytr, yte = split_data(X, y, seed=SEED); km = KMeans(2, n_init=10, random_state=SEED).fit(Xtr.values)
    out["our K-Means seed 42: cluster sizes / train positive rate / centroid ZIP,Income"] = [
        {"cluster": c, "n": int((km.labels_ == c).sum()), "pos rate": round(float(ytr.values[km.labels_ == c].mean()), 3),
         "centroid ZIP": round(float(km.cluster_centers_[c][list(X.columns).index("ZIP Code")]), 0),
         "centroid Income": round(float(km.cluster_centers_[c][list(X.columns).index("Income")]), 1)} for c in (0, 1)]
    V = {"ours: kmeans++, n_init=10, unscaled, cluster->label by train pos. rate": (False, "k-means++", 10, "train positive rate", None),
         "literal paper: init=random data points, n_init=1, unscaled": (False, "random", 1, "train positive rate", None),
         "literal paper + arbitrary mapping (cluster id 1 = class 1)": (False, "random", 1, "cluster id", None),
         "MinMax scaled, kmeans++ n_init=10": (True, "k-means++", 10, "train positive rate", None),
         "MinMax scaled, init=random n_init=1": (True, "random", 1, "train positive rate", None),
         "drop ZIP Code, unscaled": (False, "k-means++", 10, "train positive rate", "ZIP Code"),
         "drop ZIP Code + MinMax scaled": (True, "k-means++", 10, "train positive rate", "ZIP Code")}
    rows = []
    for seed in range(n_seeds):
        Xtr, Xte, ytr, yte = split_data(X, y, seed=seed)
        for name, (sc, init, ni, mp, drop) in V.items():
            a, b = (Xtr.drop(columns=[drop]), Xte.drop(columns=[drop])) if drop else (Xtr, Xte)
            rows.append({"Variant": name, "Seed": seed, **km_run(a, ytr, b, yte, sc, init, ni, mp, seed)})
    d = pd.DataFrame(rows); d.to_csv(OUT / "7_kmeans_variant_runs.csv", index=False)
    s = d.groupby("Variant", sort=False).agg(**{"AUC(score) mean": ("AUC (score)", "mean"), "AUC(score) min": ("AUC (score)", "min"), "AUC(score) max": ("AUC (score)", "max"),
        "AUC(hard) mean": ("AUC (hard labels)", "mean"), "TPR mean": ("TPR", "mean"), "FPR mean": ("FPR", "mean"), "FP mean": ("FP", "mean"), "FN mean": ("FN", "mean")}).reset_index()
    s.to_csv(OUT / "7_kmeans_variant_summary.csv", index=False); print(s.round(3).to_string())
    rows = []   # KNN hard-label operating points (ROC vertex in the figure is single => binary scores)
    for seed in range(10):
        Xtr, Xte, ytr, yte = split_data(X, y, seed=seed)
        for sc in (False, True):
            for k in (1, 3, 5, 15):
                p = make_pipeline(KNeighborsClassifier(k), sc).fit(Xtr, ytr); pr = p.predict(Xte); yt = yte.values
                rows.append({"scaled": sc, "k": k, "TPR": pr[yt == 1].mean(), "FPR": pr[yt == 0].mean(), "AUC (hard labels)": roc_auc_score(yt, pr),
                             "AUC (proba)": roc_auc_score(yt, p.predict_proba(Xte)[:, 1])})
    kn = pd.DataFrame(rows).groupby(["scaled", "k"]).mean().reset_index(); kn.to_csv(OUT / "7_knn_operating_points.csv", index=False)
    print(kn.round(3).to_string()); json.dump(out, open(OUT / "7_kmeans_analysis.json", "w"), indent=2, default=str); print(json.dumps(out, indent=1, default=str))


# ---------- items 5 & 9: MLP interpretation sensitivity; optimizer configs ----------
def opt_audit():
    X, y, Xtr, Xte, ytr, yte = _split(); rec = {}
    mk_mlp = lambda s, **kw: make_pipeline(MLPClassifier(hidden_layer_sizes=(100,), activation="relu", alpha=1e-4, max_iter=500, random_state=s, **kw), True)
    mlp = pd.read_csv(RES / "optimizer_comparison_mlp.csv").set_index("Optimizer")
    cfgs = {"adam (lr=1e-3)": dict(solver="adam", learning_rate_init=1e-3), "sgd+momentum (lr=1e-3)": dict(solver="sgd", learning_rate_init=1e-3),
            "sgd+momentum (lr=1e-2)": dict(solver="sgd", learning_rate_init=1e-2), "lbfgs": dict(solver="lbfgs")}
    rec["MLP"] = {}
    for lab, kw in cfgs.items():
        ms = []
        for s in range(5):
            m = mk_mlp(s, **kw); fit_recording(m, Xtr, ytr); met, _ = evaluate(m, Xte, yte); ms.append({**met, "n_iter": m[-1].n_iter_})
        t = pd.DataFrame(ms)
        diff = max(abs(t["PR-AUC"].mean() - mlp.loc[lab, "Test PR-AUC (mean)"]), abs(t["F1"].mean() - mlp.loc[lab, "Test F1 (mean)"]), abs(t["FP"].mean() - mlp.loc[lab, "Test FP (mean)"]))
        rec["MLP"][lab] = {"estimator params": {k: str(v) for k, v in m[-1].get_params().items()}, "scaler": "MinMaxScaler fitted on training split",
                           "init seeds (test eval)": [0, 1, 2, 3, 4], "CV": "StratifiedKFold(5, shuffle=True, random_state=42), random_state=42 estimator",
                           "max abs diff reproduced vs saved CSV (PR-AUC, F1, FP means)": float(diff),
                           "saved: iterations mean / runs hitting max_iter": [float(mlp.loc[lab, "Iterations (mean)"]), int(mlp.loc[lab, "Runs hitting max_iter / ConvergenceWarning"])]}
        print("MLP", lab, rec["MLP"][lab]["max abs diff reproduced vs saved CSV (PR-AUC, F1, FP means)"], flush=True)
    lr = pd.read_csv(RES / "optimizer_comparison_logreg.csv").set_index("Optimizer"); rec["LogReg"] = {}
    for sv in ["lbfgs", "newton-cg", "newton-cholesky", "liblinear", "sag", "saga"]:
        m = make_pipeline(LogisticRegression(solver=sv, C=1.0, l1_ratio=0.0, max_iter=2000, random_state=SEED), True); fit_recording(m, Xtr, ytr); met, _ = evaluate(m, Xte, yte)
        rec["LogReg"][sv] = {"estimator params": {k: str(v) for k, v in m[-1].get_params().items()}, "max abs diff vs saved (PR-AUC, FP, FN)":
                             float(max(abs(met["PR-AUC"] - lr.loc[sv, "Test PR-AUC"]), abs(met["FP"] - lr.loc[sv, "Test FP"]), abs(met["FN"] - lr.loc[sv, "Test FN"])))}
    pc = pd.read_csv(RES / "optimizer_comparison_perceptron.csv").set_index("Optimizer"); rec["Perceptron"] = {}
    for pen in [None, "l2", "l1", "elasticnet"]:
        for eta0 in [0.1, 1.0]:
            m = make_pipeline(Perceptron(penalty=pen, alpha=1e-4, eta0=eta0, max_iter=1000, random_state=SEED), True); fit_recording(m, Xtr, ytr); met, _ = evaluate(m, Xte, yte)
            lab = f"penalty={pen}, eta0={eta0}"
            rec["Perceptron"][lab] = {"estimator params": {k: str(v) for k, v in m[-1].get_params().items()}, "max abs diff vs saved (PR-AUC, FP, FN)":
                                      float(max(abs(met["PR-AUC"] - pc.loc[lab, "Test PR-AUC"]), abs(met["FP"] - pc.loc[lab, "Test FP"]), abs(met["FN"] - pc.loc[lab, "Test FN"])))}
    # liblinear vs lbfgs: does intercept regularisation explain the small gap?  and l1_ratio=0 == penalty='l2'?
    sc = MinMaxScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    ref = LogisticRegression(solver="lbfgs", max_iter=2000).fit(A, ytr).decision_function(B); chk = {}
    for isc in (1, 1e3, 1e6):
        d = LogisticRegression(solver="liblinear", intercept_scaling=isc, max_iter=2000).fit(A, ytr).decision_function(B); chk[f"liblinear intercept_scaling={isc:g}: max|decision_function - lbfgs|"] = float(np.max(np.abs(d - ref)))
    for sv in ("lbfgs", "liblinear", "saga"):
        a = LogisticRegression(solver=sv, l1_ratio=0.0, max_iter=5000).fit(A, ytr).coef_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); b = LogisticRegression(solver=sv, penalty="l2", max_iter=5000).fit(A, ytr).coef_
        chk[f"{sv}: l1_ratio=0.0 vs penalty='l2' max|coef diff|"] = float(np.max(np.abs(a - b)))
    rec["LogReg solver checks"] = chk
    # item 5: sensitivity of the "100 hidden layers" interpretation (5 init seeds, scaled, paper MLP settings otherwise)
    rows = []
    for hl in [(100,), (100, 100), (100, 100, 100)]:
        for s in range(5):
            m = make_pipeline(MLPClassifier(hidden_layer_sizes=hl, activation="relu", solver="adam", alpha=1e-4, learning_rate="constant", random_state=s), True)
            fit_recording(m, Xtr, ytr); met, _ = evaluate(m, Xte, yte); rows.append({"hidden_layer_sizes": str(hl), "seed": s, **met})
    h = pd.DataFrame(rows).groupby("hidden_layer_sizes")[["Accuracy", "Recall", "F1", "ROC-AUC", "PR-AUC", "FP", "FN"]].agg(["mean", "std"]); h.to_csv(OUT / "5_mlp_hidden_layer_interpretations.csv")
    d0 = MLPClassifier().get_params(); rec["sklearn MLPClassifier defaults matching every setting the paper lists"] = {k: str(d0[k]) for k in ("hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate")}
    json.dump(rec, open(OUT / "9_optimizer_exact_configs.json", "w"), indent=2); print(h.round(3).to_string()); print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    sec = sys.argv[1]
    if sec == "quick": check_params_and_scaling(); check_metrics(); check_mlp_search(); check_no_test_in_selection()
    elif sec == "lr": lr_study()
    elif sec == "km": km_study()
    elif sec == "opt": opt_audit()
