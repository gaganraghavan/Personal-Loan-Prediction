"""End-to-end pipeline:  python -m src.experiments   (from the project root)

BASELINE -> HYPERPARAMETER TUNING (train only, CV) -> TUNED MODEL -> FINAL TEST EVALUATION
"""
import json
import pickle
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from . import plots
from .baseline import run_baselines, seed_variability
from .evaluation import evaluate, fit_recording
from .optimizers import compare_logreg, compare_mlp, compare_perceptron
from .preprocessing import SEED, load_data, split_data
from .tuning import run_search, search_spaces

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"; PLOTS = RES / "plots"; CVDIR = RES / "cv_results"
MODELS = ["Perceptron", "Logistic Regression", "K-Means", "KNeighbors", "SVM", "MLP"]
TUNED_MODELS = ["Perceptron", "Logistic Regression", "KNeighbors", "SVM", "MLP"]  # K-Means: nothing meaningful to tune (k=2 fixed by the paper's design)
METRICS = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "Balanced Accuracy", "ROC-AUC", "PR-AUC", "FP", "FN"]

# Numbers as REPORTED BY THE PAPER (Sec. 5 text + ROC legend). Not produced by us.
PAPER = [
    {"Model": "Perceptron", "Paper AUC": 0.61, "Paper note": "predicted all zeros (text)"},
    {"Model": "Logistic Regression", "Paper AUC": 0.81, "Paper note": "predicted all zeros (text)"},
    {"Model": "K-Means", "Paper AUC": 0.75, "Paper FP": 7, "Paper FN": 72, "Paper Accuracy": 0.90, "Paper note": "text says K-Means, AUC .75, 7 FP / 72 FN"},
    {"Model": "KNeighbors", "Paper AUC": 0.75, "Paper note": "ROC legend label 'KNeighborsClassifier' AUC .75"},
    {"Model": "SVM", "Paper AUC": 0.94, "Paper FP": 27, "Paper FN": 56, "Paper note": "linear kernel"},
    {"Model": "MLP", "Paper AUC": 0.99, "Paper FP": 9, "Paper FN": 11, "Paper Accuracy": 0.98, "Paper note": "after MinMax scaling; before scaling ~.93 acc / .93 AUC"},
]


def sensitivity_table(cv_tables):
    rows = []
    for m, df in cv_tables.items():
        for c in [c for c in df.columns if c.startswith("param_clf__")]:
            g = df.assign(v=df[c].astype(str)).groupby("v")["mean_test_average_precision"]
            means = g.mean()
            if len(means) < 2: continue
            rows.append({"Model": m, "Hyperparameter": c.replace("param_clf__", ""), "n values": len(means),
                         "Marginal range of mean CV PR-AUC": means.max() - means.min(),
                         "Best value (by marginal mean)": means.idxmax()})
    s = pd.DataFrame(rows)
    s["Overall CV PR-AUC range (all configs)"] = s["Model"].map({m: d["mean_test_average_precision"].max() - d["mean_test_average_precision"].min() for m, d in cv_tables.items()})
    return s


def main():
    t0 = time.time()
    for d in (RES, PLOTS, CVDIR): d.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    Xtr, Xte, ytr, yte = split_data(X, y, seed=SEED)
    print(f"data {X.shape}, train {Xtr.shape}, test {Xte.shape}; positives train={ytr.sum()} test={yte.sum()}")
    plots.class_distribution(y, ytr, yte, PLOTS / "01_class_distribution.png")

    # ---------------- Phase 3: baselines ----------------
    base_df, preds, hp = run_baselines(Xtr, ytr, Xte, yte, SEED)
    base_df.to_csv(RES / "baseline_results.csv", index=False)
    print(base_df[["Model", "Version"] + METRICS].round(3).to_string()); print(f"[{time.time()-t0:.0f}s] baselines done")

    # ---------------- Phase 5: tuning (train only) ----------------
    tuned_rows, cv_tables, best = [], {}, {}
    for name, (kind, est, space) in search_spaces(SEED).items():
        gs, cvdf, secs = run_search(name, kind, est, space, Xtr, ytr, SEED)
        cvdf.to_csv(CVDIR / f"{name.replace(' ', '_')}_cv_results.csv", index=False)
        cv_tables[name] = cvdf
        model = gs.best_estimator_            # already refit on the full TRAINING split
        met, p = evaluate(model, Xte, yte)    # first and only contact with the test set
        bi = gs.best_index_
        tuned_rows.append({"Model": name, "Version": "Tuned", **met,
                           "Best CV PR-AUC": gs.best_score_,
                           "CV Recall at best": cvdf.loc[bi, "mean_test_recall"], "CV F1 at best": cvdf.loc[bi, "mean_test_f1"],
                           "CV ROC-AUC at best": cvdf.loc[bi, "mean_test_roc_auc"],
                           "Configs evaluated": len(cvdf), "Search type": kind, "Search seconds": round(secs, 1),
                           "Best params": json.dumps({k.replace("clf__", ""): str(v) for k, v in gs.best_params_.items()})})
        preds[(name, "Tuned")] = p; best[name] = {k.replace("clf__", ""): str(v) for k, v in gs.best_params_.items()}
        hp[f"{name} | Tuned"] = {k: str(v) for k, v in model.get_params(deep=True).items() if "__" in k}
        print(f"[{time.time()-t0:.0f}s] tuned {name}: CV PR-AUC {gs.best_score_:.3f}  best={best[name]}")
        if name == "MLP": tuned_mlp = model
    tuned_df = pd.DataFrame(tuned_rows); tuned_df.to_csv(RES / "tuned_results.csv", index=False)
    json.dump(best, open(RES / "best_hyperparameters.json", "w"), indent=2)
    json.dump(hp, open(RES / "final_model_hyperparameters_all.json", "w"), indent=2)

    # ---------------- Phase 7: comparison ----------------
    comp = pd.concat([base_df, tuned_df], ignore_index=True)
    comp["Version"] = pd.Categorical(comp["Version"], plots.VERSIONS, ordered=True)
    comp["Model"] = pd.Categorical(comp["Model"], MODELS, ordered=True)
    comp = comp.sort_values(["Model", "Version"])
    cols = ["Model", "Version"] + METRICS + ["ROC-AUC"]
    comp[["Model", "Version"] + METRICS + ["TP", "TN"]].to_csv(RES / "model_comparison.csv", index=False)
    paper = pd.DataFrame(PAPER).merge(base_df[base_df.Version == "Baseline (paper-faithful)"][["Model", "ROC-AUC", "Accuracy", "FP", "FN"]]
                                      .rename(columns={"ROC-AUC": "Ours ROC-AUC", "Accuracy": "Ours Accuracy", "FP": "Ours FP", "FN": "Ours FN"}), on="Model", how="left")
    paper.to_csv(RES / "paper_vs_replication.csv", index=False)
    print(comp[["Model", "Version"] + METRICS].round(3).to_string())

    # ---------------- Phase 6: optimizers ----------------
    mlp_opt = compare_mlp(Xtr, ytr, Xte, yte, SEED); mlp_opt.to_csv(RES / "optimizer_comparison_mlp.csv", index=False)
    lr_opt = compare_logreg(Xtr, ytr, Xte, yte, SEED); lr_opt.to_csv(RES / "optimizer_comparison_logreg.csv", index=False)
    pc_opt = compare_perceptron(Xtr, ytr, Xte, yte, SEED); pc_opt.to_csv(RES / "optimizer_comparison_perceptron.csv", index=False)
    print(mlp_opt.round(3).to_string()); print(lr_opt.round(4).to_string()); print(f"[{time.time()-t0:.0f}s] optimizers done")

    # ---------------- robustness: seeds ----------------
    sv_df, sv_agg = seed_variability(range(10)); sv_df.to_csv(RES / "baseline_seed_runs.csv", index=False)
    sv_agg.to_csv(RES / "baseline_seed_variability_summary.csv", index=False)
    rows = []
    for s in range(10):  # MLP init-seed sensitivity at FIXED hyperparameters, same split
        from .models import baseline_estimators
        b = baseline_estimators(s, False)["MLP"]; fit_recording(b, Xtr, ytr); mb, _ = evaluate(b, Xte, yte)
        t = clone(tuned_mlp).set_params(clf__random_state=s); fit_recording(t, Xtr, ytr); mt, _ = evaluate(t, Xte, yte)
        rows += [{"Version": "Baseline (paper-faithful)", "Init seed": s, **mb}, {"Version": "Tuned", "Init seed": s, **mt}]
    mi = pd.DataFrame(rows); mi.to_csv(RES / "mlp_init_seed_runs.csv", index=False)
    mi.groupby("Version")[["Accuracy", "Recall", "F1", "ROC-AUC", "PR-AUC", "FP", "FN"]].agg(["mean", "std"]).to_csv(RES / "mlp_init_seed_summary.csv")

    # ---------------- sensitivity + plots ----------------
    sens = sensitivity_table(cv_tables); sens.to_csv(RES / "hyperparameter_sensitivity.csv", index=False)
    plots.confusion_grid(preds, MODELS, PLOTS / "02_confusion_matrices.png")
    plots.curves(preds, MODELS, "roc", PLOTS / "03_roc_curves.png")
    plots.curves(preds, MODELS, "pr", PLOTS / "04_precision_recall_curves.png")
    plots.baseline_vs_tuned(comp, PLOTS / "05_baseline_vs_tuned_metrics.png")
    plots.optimizer_plot(mlp_opt, lr_opt, PLOTS / "06_optimizer_comparison.png")
    plots.sensitivity_plot(cv_tables, sens, PLOTS / "07_hyperparameter_sensitivity.png")
    pickle.dump(preds, open(RES / "predictions.pkl", "wb"))
    from .ensembles import run_ensemble_extension   # NEW EXTENSION (not in the paper); writes new files only
    run_ensemble_extension()
    print(f"ALL DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    import sys
    if "--ensemble-only" in sys.argv:   # run just the ensemble extension from the saved results (existing outputs untouched)
        from .ensembles import run_ensemble_extension
        n = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 5
        run_ensemble_extension(n_seed_runs=n)
    else:
        main()
