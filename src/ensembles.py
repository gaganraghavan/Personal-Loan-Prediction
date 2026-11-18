"""ENSEMBLE LEARNING EXTENSION -- NOT part of the original paper.

Combines the already-tuned Logistic Regression, SVM and MLP (hyperparameters read from
results/final_model_hyperparameters_all.json; NO new hyperparameter search) with
  A) soft voting (mean of predicted probabilities, equal weights)
  B) stacking (out-of-fold probabilities -> Logistic Regression meta-model).

Leakage control: every component is a Pipeline(MinMaxScaler -> estimator), so scaling is fitted inside
each fit / internal-CV training fold. `fit_ensemble` only receives the TRAINING split; the test split
is touched only by `evaluate(...)` after fitting. Same split (seed 42), same 0.5 threshold, same
evaluation.py metrics as the rest of the project. Existing result files are never overwritten.

Run:  python -m src.experiments --ensemble-only   (or as the last stage of the full pipeline)
"""
import ast
import json
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from . import plots
from .evaluation import evaluate, fit_recording
from .preprocessing import SEED, load_data, make_pipeline, split_data

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"; PLOTS = RES / "plots"
BASE_MODELS = ["Logistic Regression", "SVM", "MLP"]       # K-Means: no probabilities; Perceptron/KNN: not used (see README)
_CLS = {"Logistic Regression": LogisticRegression, "SVM": SVC, "MLP": MLPClassifier}
SOFT, STACK = "Soft Voting Ensemble", "Stacking Ensemble"
VER_SOFT, VER_STACK = "Extension - Soft Voting", "Extension - Stacking"
METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "Balanced Accuracy", "ROC-AUC", "PR-AUC", "FP", "FN", "TP", "TN"]
STACK_CV_FOLDS = 5


# ----------------------------------------------------------------- hyperparameters (reused, not re-tuned)
def _parse(v):
    try: return ast.literal_eval(v)
    except (ValueError, SyntaxError): return v


def load_tuned_params(path=RES / "final_model_hyperparameters_all.json"):
    """Parse the saved (string-valued) final tuned estimator parameters for LR / SVM / MLP."""
    j = json.load(open(path)); out = {}
    for m in BASE_MODELS:
        d = j[f"{m} | Tuned"]
        # the saved scaler must be the default MinMaxScaler that make_pipeline() uses
        assert d["scaler__feature_range"] == "(0, 1)" and d["scaler__clip"] == "False", d
        out[m] = {k[len("clf__"):]: _parse(v) for k, v in d.items() if k.startswith("clf__")}
    return out


def make_component(name, params, seed=SEED, probability=False):
    """Scaler->estimator pipeline with the tuned hyperparameters.
    random_state is set to `seed` (equal to the saved 42 for LR/MLP; for SVC it only controls the
    internal Platt-scaling CV and is needed when probability=True)."""
    p = dict(params); p["random_state"] = seed
    if name == "SVM": p["probability"] = probability   # probability=True ONLY for the ensemble extension
    return make_pipeline(_CLS[name](**p), scaled=True)


# ----------------------------------------------------------------- ensemble definitions
def _estimators(params, seed):
    return [("lr", make_component("Logistic Regression", params["Logistic Regression"], seed)),
            ("svm", make_component("SVM", params["SVM"], seed, probability=True)),
            ("mlp", make_component("MLP", params["MLP"], seed))]


def make_soft_voting_ensemble(params, seed=SEED):
    return VotingClassifier(estimators=_estimators(params, seed), voting="soft")   # equal weights


def make_stacking_ensemble(params, seed=SEED):
    """Meta-model: Logistic Regression (default C=1, NOT tuned) on the out-of-fold predicted
    probabilities (5-fold StratifiedKFold on the training split only); base models are then refit on all training rows."""
    cv = StratifiedKFold(STACK_CV_FOLDS, shuffle=True, random_state=seed)
    return StackingClassifier(estimators=_estimators(params, seed),
                              final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
                              cv=cv, stack_method="predict_proba", passthrough=False)


def fit_ensemble(model, Xtr, ytr):
    """Only the TRAINING split can reach this function."""
    return fit_recording(model, Xtr, ytr)


# ----------------------------------------------------------------- sanity checks
def sanity_checks(params, voting, stacking, comp_models, preds_saved, tuned_df, Xtr, ytr, Xte):
    chk = {}
    saved = pd.read_csv(RES / "tuned_results.csv").set_index("Model")
    chk["tuned params saved with random_state 42 for LR and MLP (== SEED)"] = bool(params["Logistic Regression"]["random_state"] == SEED == params["MLP"]["random_state"])
    # 1. reconstructed components reproduce the existing tuned results exactly
    for name, res in tuned_df.items():
        chk[f"reconstructed {name} (tuned) vs tuned_results.csv: max abs metric diff"] = float(
            max(abs(res[k] - saved.loc[name, k]) for k in METRIC_COLS))
    # 2. probability-enabled SVM gives the same hard labels as the existing tuned SVM
    chk["SVM probability=True: hard labels identical to existing tuned SVM"] = bool(
        (comp_models["SVM"].predict(Xte) == preds_saved[("SVM", "Tuned")]["y_pred"]).all())
    # 3. soft voting = mean of the component probabilities, label = (mean > 0.5)
    mean_p = np.mean([e.predict_proba(Xte)[:, 1] for e in voting.estimators_], axis=0)
    chk["voting: predict_proba == mean of component probabilities (max abs diff)"] = float(np.max(np.abs(voting.predict_proba(Xte)[:, 1] - mean_p)))
    chk["voting: predict == (mean probability > 0.5)"] = bool(((mean_p > 0.5).astype(int) == voting.predict(Xte)).all())
    # 4. stacking meta-model is trained on out-of-fold TRAIN predictions only
    cv = StratifiedKFold(STACK_CV_FOLDS, shuffle=True, random_state=SEED)
    Z = np.column_stack([cross_val_predict(clone(est), Xtr, ytr, cv=cv, method="predict_proba")[:, 1] for _, est in _estimators(params, SEED)])
    meta = LogisticRegression(max_iter=1000, random_state=SEED).fit(Z, ytr)
    chk["stacking: meta coef == manual OOF(train-only) refit, max abs diff"] = float(np.max(np.abs(meta.coef_ - stacking.final_estimator_.coef_)))
    chk["stacking: meta-model n_features (3 base models, passthrough=False)"] = int(stacking.final_estimator_.n_features_in_)
    chk["stacking: meta coefficients [LR, SVM, MLP]"] = [round(float(c), 4) for c in stacking.final_estimator_.coef_[0]]
    chk["stacking: meta intercept"] = round(float(stacking.final_estimator_.intercept_[0]), 4)
    return chk


# ----------------------------------------------------------------- seed-robustness (supplementary)
def seed_robustness(params, Xtr, ytr, Xte, yte, seeds):
    """Supplementary: SAME split, only the stochastic components (MLP init, Platt CV, stacking CV folds) re-seeded."""
    rows = []
    for s in seeds:
        ms = {"Logistic Regression (tuned)": make_component("Logistic Regression", params["Logistic Regression"], s),
              "SVM (tuned)": make_component("SVM", params["SVM"], s),
              "MLP (tuned)": make_component("MLP", params["MLP"], s),
              SOFT: make_soft_voting_ensemble(params, s), STACK: make_stacking_ensemble(params, s)}
        for n, m in ms.items():
            fit_recording(m, Xtr, ytr); met, _ = evaluate(m, Xte, yte); rows.append({"Model": n, "Seed": s, **met})
        print(f"  seed {s} done", flush=True)
    d = pd.DataFrame(rows); d.to_csv(RES / "ensemble_seed_runs.csv", index=False)
    cols = ["Precision", "Recall", "F1", "Balanced Accuracy", "ROC-AUC", "PR-AUC", "FP", "FN"]
    g = d.groupby("Model", sort=False)[cols].agg(["mean", "std"]); g.columns = [f"{a} ({b})" for a, b in g.columns]
    g = g.reset_index(); g.insert(1, "n seeds", len(list(seeds))); g.to_csv(RES / "ensemble_seed_summary.csv", index=False)
    return g


# ----------------------------------------------------------------- main entry point
def run_ensemble_extension(n_seed_runs=5, checks=True):
    warnings.filterwarnings("ignore")
    X, y = load_data(); Xtr, Xte, ytr, yte = split_data(X, y, seed=SEED)        # identical split to the rest of the project
    params = load_tuned_params()
    tuned = pd.read_csv(RES / "tuned_results.csv").set_index("Model")
    preds_saved = pickle.load(open(RES / "predictions.pkl", "rb"))
    best_ind = max(BASE_MODELS, key=lambda m: tuned.loc[m, "Best CV PR-AUC"])    # chosen from TRAIN-ONLY CV, not from test results
    print("best individual by train-only CV PR-AUC:", best_ind, {m: round(tuned.loc[m, "Best CV PR-AUC"], 4) for m in BASE_MODELS})

    # components as used inside the ensembles (reference + sanity check)
    comp, comp_rows, comp_res = {}, [], {}
    for n in BASE_MODELS:
        m = make_component(n, params[n], SEED, probability=(n == "SVM")); secs, conv = fit_ensemble(m, Xtr, ytr)
        met, _ = evaluate(m, Xte, yte); comp[n] = m; comp_res[n] = met
        comp_rows.append({"Model": n, "Version": "Component (ensemble config" + ("; SVM probability=True)" if n == "SVM" else ")"), **met, "Fit seconds": round(secs, 1)})
    pd.DataFrame(comp_rows).to_csv(RES / "ensemble_components.csv", index=False)

    models = {SOFT: (VER_SOFT, make_soft_voting_ensemble(params), "mean of predicted probabilities (equal weights)"),
              STACK: (VER_STACK, make_stacking_ensemble(params), f"LogisticRegression meta-model on {STACK_CV_FOLDS}-fold out-of-fold probabilities")}
    rows, preds = [], {}
    for name, (ver, model, how) in models.items():
        secs, conv = fit_ensemble(model, Xtr, ytr); met, p = evaluate(model, Xte, yte)
        rows.append({"Model": name, "Version": ver, **{k: met[k] for k in METRIC_COLS}, "Predicted positives": met["Predicted positives"],
                     "Fit seconds": round(secs, 1), "Base models": "LR + SVM(probability=True) + MLP (tuned)", "Combination": how, "Seed": SEED})
        preds[name] = p; print(f"{name}: F1={met['F1']:.3f} recall={met['Recall']:.3f} PR-AUC={met['PR-AUC']:.3f} ROC-AUC={met['ROC-AUC']:.3f} FP={met['FP']} FN={met['FN']}", flush=True)
    ens = pd.DataFrame(rows); ens.to_csv(RES / "ensemble_results.csv", index=False)

    # ---- comparison tables (existing files are left untouched; new files only)
    existing = pd.read_csv(RES / "model_comparison.csv")
    origin = {"Baseline (paper-faithful)": "Paper replication", "Baseline (all scaled)": "Control (our)", "Tuned": "Extension - tuning"}
    ext = existing.assign(Origin=existing["Version"].map(origin))
    new = ens[["Model", "Version"] + METRIC_COLS[:10] + ["TP", "TN"]].assign(Origin="Extension - ensemble")
    pd.concat([ext, new], ignore_index=True).to_csv(RES / "model_comparison_with_ensembles.csv", index=False)

    ind = []
    for n in BASE_MODELS:
        r = tuned.loc[n]; ind.append({"Model": n + " (tuned)", "Version": "Tuned", **{k: r[k] for k in METRIC_COLS}, "Train-only CV PR-AUC": r["Best CV PR-AUC"], "Best individual (by CV PR-AUC)": n == best_ind})
    cmp_df = pd.concat([pd.DataFrame(ind), ens.assign(Model=ens["Model"])[["Model", "Version"] + METRIC_COLS]], ignore_index=True)
    cmp_df["Best individual (by CV PR-AUC)"] = cmp_df["Best individual (by CV PR-AUC)"].fillna(False)
    cmp_df.to_csv(RES / "ensemble_comparison.csv", index=False)

    drows = []
    for _, e in ens.iterrows():
        for k in ["F1", "Recall", "Precision", "PR-AUC", "ROC-AUC", "Balanced Accuracy", "Accuracy", "FP", "FN"]:
            b = tuned.loc[best_ind, k]
            drows.append({"Ensemble": e["Model"], "Metric": k, "Ensemble value": e[k], f"Best individual by CV ({best_ind})": b, "Delta (ensemble - best individual)": e[k] - b,
                          **{n: tuned.loc[n, k] for n in BASE_MODELS}})
    pd.DataFrame(drows).to_csv(RES / "ensemble_vs_best_individual.csv", index=False)

    # ---- plots (new files; existing plots untouched)
    labels = {f"{n} (tuned)": preds_saved[(n, "Tuned")] for n in BASE_MODELS}; labels.update({SOFT: preds[SOFT], STACK: preds[STACK]})
    pickle.dump(labels, open(RES / "ensemble_predictions.pkl", "wb"))
    plots.ensemble_metrics(cmp_df, f"{best_ind} (tuned)", PLOTS / "08_ensemble_metrics.png")
    plots.ensemble_curves(labels, PLOTS / "09_ensemble_roc_pr_curves.png")
    plots.ensemble_confusions(labels, PLOTS / "10_ensemble_confusion_matrices.png")

    if checks:
        chk = sanity_checks(params, models[SOFT][1], models[STACK][1], comp, preds_saved, comp_res_tuned(comp_res, params, Xtr, ytr, Xte, yte), Xtr, ytr, Xte)
        chk["best individual by train-only CV PR-AUC"] = best_ind
        json.dump(chk, open(RES / "ensemble_sanity_checks.json", "w"), indent=2); print(json.dumps(chk, indent=1))
    if n_seed_runs:
        print(f"seed robustness over seeds 0..{n_seed_runs-1}"); print(seed_robustness(params, Xtr, ytr, Xte, yte, range(n_seed_runs)).round(3).to_string())
    return ens


def comp_res_tuned(comp_res, params, Xtr, ytr, Xte, yte):
    """Metrics of the components exactly as in the existing tuned experiment (SVM with probability=False)."""
    out = {}
    for n in BASE_MODELS:
        m = make_component(n, params[n], SEED, probability=False); fit_recording(m, Xtr, ytr); out[n], _ = evaluate(m, Xte, yte)
    return out
