"""Tables, selection rules and plots for the synthetic-data experiment (reads only saved CSVs/pickles)."""
import json, pickle
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from .preprocessing import load_data, split_data
from .synthetic_data import TARGET

sns.set_theme(style="whitegrid", context="notebook")
MODELS = ["Perceptron", "Logistic Regression", "K-Means", "KNN", "SVM", "MLP"]
COL = dict(zip(MODELS, ["#9aa5b1", "#4c78a8", "#b279a2", "#72b7b2", "#f58518", "#54a24b"]))
PERF_METRICS = ["F1", "Balanced Accuracy", "ROC-AUC", "PR-AUC"]      # 'PR-AUC' = average precision
EFF = ["Tuning time (s)", "Training time (s)", "Prediction time (s, test set)", "Inference per record (us, batch avg)", "Single-record latency (us, median)", "Model size (KB)"]
ROOT = Path(__file__).resolve().parents[1]


def _save(fig, path): fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def _ms(df, cols, keys=("Model",)):
    g = df.groupby(list(keys), sort=False)[cols].agg(["mean", "std"]); g.columns = [f"{a} ({b})" for a, b in g.columns]
    return g.reset_index().set_index("Model").reindex(MODELS).reset_index()


def _md(df, floatfmt="{:.3f}"):
    cells = lambda r: [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(cells(r)) + " |" for r in df.to_numpy()])


def pareto(df, up, down):
    keep = []
    for i, r in df.iterrows():
        better = (df[up] >= r[up]) & (df[down] <= r[down]).all(axis=1) & ((df[up] > r[up]) | (df[down] < r[down]).any(axis=1))
        keep.append(not better.any())
    return keep


def build_report(out):
    T = out / "tables"; P = out / "plots"
    perf = pd.read_csv(T / "per_seed_performance.csv"); eff = pd.read_csv(T / "per_seed_efficiency.csv"); qual = pd.read_csv(T / "quality_per_seed.csv")
    n_seeds = perf["Seed"].nunique(); allm = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "Balanced Accuracy", "ROC-AUC", "PR-AUC", "TP", "FP", "TN", "FN"]
    syn, real = perf[perf["Eval set"] == "Synthetic test"].copy(), perf[perf["Eval set"] == "Real test"].copy()
    syn["Performance score"] = syn[PERF_METRICS].mean(axis=1)                      # mean of F1, BalAcc, ROC-AUC, PR-AUC (per seed)
    eff["Tuning + training time (s)"] = eff["Tuning time (s)"] + eff["Training time (s)"]
    sm = _ms(syn, allm + ["Performance score"]); sm.to_csv(T / "performance_summary_synthetic_test.csv", index=False)
    _ms(real, allm).to_csv(T / "performance_summary_real_test_TSTR.csv", index=False)
    em = _ms(eff, EFF + ["Tuning + training time (s)"]); em.to_csv(T / "efficiency_summary.csv", index=False)
    _ms(syn, ["TP", "FP", "TN", "FN"]).to_csv(T / "confusion_matrix_counts_synthetic_test.csv", index=False)
    q = qual.drop(columns=["Seed", "Generator acceptance rate (class 0 / 1)"]); qs = pd.DataFrame({"mean": q.mean(), "std": q.std()}); qs.to_csv(T / "quality_summary.csv")
    # compact final table
    fc = sm[["Model"] + [f"{m} (mean)" for m in ["Performance score", "F1", "Balanced Accuracy", "ROC-AUC", "PR-AUC"]] + ["Performance score (std)"]].copy()
    fc = fc.merge(em[["Model", "Tuning time (s) (mean)", "Training time (s) (mean)", "Tuning + training time (s) (mean)", "Inference per record (us, batch avg) (mean)", "Model size (KB) (mean)"]], on="Model")
    fc = fc.merge(_ms(real, ["ROC-AUC"])[["Model", "ROC-AUC (mean)"]].rename(columns={"ROC-AUC (mean)": "Real-test ROC-AUC (mean, train-on-synthetic)"}), on="Model")
    fc.to_csv(T / "final_comparison.csv", index=False)

    # ---- real-trained reference (existing project, seed 42, same real test split)
    try:
        rr = pd.read_csv(ROOT / "results" / "tuned_results.csv").set_index("Model"); bb = pd.read_csv(ROOT / "results" / "baseline_results.csv")
        ref = {"Perceptron": rr.loc["Perceptron"], "Logistic Regression": rr.loc["Logistic Regression"], "KNN": rr.loc["KNeighbors"], "SVM": rr.loc["SVM"], "MLP": rr.loc["MLP"],
               "K-Means": bb[(bb.Model == "K-Means") & (bb.Version == "Baseline (all scaled)")].iloc[0]}
        r42 = real[real.Seed == 42].set_index("Model")
        rows = [{"Model": m, **{f"Synthetic-trained {k}": r42.loc[m, k] for k in ["ROC-AUC", "PR-AUC", "Recall", "F1", "FP", "FN"]},
                **{f"Real-trained {k}": ref[m][k] for k in ["ROC-AUC", "PR-AUC", "Recall", "F1", "FP", "FN"]}} for m in MODELS]
        pd.DataFrame(rows).to_csv(T / "real_trained_reference_seed42.csv", index=False)
    except Exception as e: print("reference skipped:", e)

    # ---- pre-declared selection rules
    m = sm.set_index("Model"); e = em.set_index("Model")
    best_perf = m["Performance score (mean)"].idxmax(); srt = m["Performance score (mean)"].sort_values(ascending=False)
    gap = srt.iloc[0] - srt.iloc[1]; maxstd = max(m.loc[srt.index[0], "Performance score (std)"], m.loc[srt.index[1], "Performance score (std)"])
    cand = [k for k in MODELS if m.loc[k, "Performance score (mean)"] >= srt.iloc[0] - 0.01]
    balance = min(cand, key=lambda k: e.loc[k, "Tuning + training time (s) (mean)"])
    pf = pd.DataFrame({"Model": MODELS, "perf": m["Performance score (mean)"].values, "cost": e["Tuning + training time (s) (mean)"].values, "inf": e["Inference per record (us, batch avg) (mean)"].values})
    pf["pareto"] = pareto(pf, "perf", ["cost", "inf"])
    concl = {"seeds": int(n_seeds), "performance_score_definition": "mean of F1, Balanced Accuracy, ROC-AUC, PR-AUC(average precision) on the synthetic test set",
             "best_predictive_performance": best_perf, "gap_to_second": round(float(gap), 4), "second": srt.index[1], "gap_smaller_than_seed_std": bool(gap < maxstd),
             "per_metric_winners": {k: m[f"{k} (mean)"].idxmax() for k in PERF_METRICS + ["Recall", "Precision", "Accuracy"]},
             "fastest_to_train_final_fit": e["Training time (s) (mean)"].idxmin(), "fastest_total_tuning_plus_training": e["Tuning + training time (s) (mean)"].idxmin(),
             "fastest_inference_per_record": e["Inference per record (us, batch avg) (mean)"].idxmin(), "smallest_model": e["Model size (KB) (mean)"].idxmin(),
             "best_balance_rule": "among models with performance score within 0.01 of the best, the lowest mean tuning+training time",
             "best_balance_candidates": cand, "best_balance": balance, "pareto_front(perf up, tune+train time down, inference down)": list(pf[pf.pareto].Model)}
    json.dump(concl, open(T / "conclusions.json", "w"), indent=1)
    disp = fc.rename(columns=lambda c: c.replace(" (mean)", "")).copy()
    open(T / "conclusions.md", "w").write("# Synthetic balanced experiment: summary (mean over %d seeds)\n\n" % n_seeds + _md(disp, "{:.4g}") + "\n\n```json\n" + json.dumps(concl, indent=1) + "\n```\n")

    # ---- plots: data quality (seed 42)
    X, y = load_data(); Xtr, _, ytr, _ = split_data(X, y, seed=42); rt = Xtr.assign(**{TARGET: ytr}); s = pd.read_csv(out / "data" / "synthetic_balanced_seed42.csv")
    feats = ["Income", "CCAvg", "Mortgage", "Age"]; fig, ax = plt.subplots(2, 4, figsize=(15, 6))
    for i, c in enumerate((0, 1)):
        for j, f in enumerate(feats):
            bins = np.histogram_bin_edges(rt[f], 30); ax[i, j].hist(rt[rt[TARGET] == c][f], bins, density=True, alpha=.55, color="#9aa5b1", label="real train")
            ax[i, j].hist(s[s[TARGET] == c][f], bins, density=True, alpha=.55, color="#e45756", label="synthetic"); ax[i, j].set_title(f"{f} | class {c}", fontsize=10)
    ax[0, 0].legend(fontsize=8); fig.suptitle("Real vs synthetic marginal distributions by class (seed 42)"); _save(fig, P / "01_synthetic_quality_distributions.png")
    cols = [c for c in rt.columns if c != TARGET]; fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for i, c in enumerate((1, 0)):
        R = rt[rt[TARGET] == c][cols].corr(method="spearman"); S = s[s[TARGET] == c][cols].corr(method="spearman")
        for a, M, t, kw in [(ax[i, 0], R, "real train", dict(vmin=-1, vmax=1, cmap="coolwarm")), (ax[i, 1], S, "synthetic", dict(vmin=-1, vmax=1, cmap="coolwarm")), (ax[i, 2], (R - S).abs(), "absolute difference", dict(vmin=0, vmax=.5, cmap="Reds"))]:
            sns.heatmap(M, ax=a, cbar=False, xticklabels=[x[:6] for x in cols], yticklabels=[x[:6] for x in cols] if a is ax[i, 0] else False, **kw); a.set_title(f"Spearman, class {c}: {t}", fontsize=10)
    _save(fig, P / "02_synthetic_quality_correlations.png")
    # ---- performance
    def bars(ax, df, col, title, log=False, lo=None):
        mu, sd = df[f"{col} (mean)"].to_numpy(float), df[f"{col} (std)"].fillna(0).to_numpy(float)
        ax.bar(range(len(MODELS)), mu, yerr=sd, capsize=3, color=[COL[k] for k in MODELS])
        for i, v in enumerate(mu):
            if v > 0 or not log: ax.text(i, v + sd[i], f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, rotation=25, ha="right", fontsize=9); ax.set_title(title)
        if log: ax.set_yscale("log")
        if lo is not None: ax.set_ylim(lo, min(1.0, mu.max() + sd.max() + .06))
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    for a, k in zip(ax.ravel(), PERF_METRICS):
        lo = max(0, float((sm[f"{k} (mean)"] - sm[f"{k} (std)"].fillna(0)).min()) - .06); bars(a, sm, k, f"{k}" + (" (average precision)" if k == "PR-AUC" else "") + ("  [axis truncated]" if lo > 0 else ""), lo=lo)
    fig.suptitle(f"Predictive performance on the synthetic balanced test set (mean ± sd over {n_seeds} seeds)"); _save(fig, P / "03_predictive_performance.png")
    # ---- efficiency
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    for a, k in zip(ax.ravel(), ["Tuning time (s)", "Training time (s)", "Inference per record (us, batch avg)", "Model size (KB)"]):
        bars(a, em, k, k + "  [log scale]", log=True)
    ax[0, 0].text(0.5, 0.92, "K-Means: not tuned", transform=ax[0, 0].transAxes, fontsize=8, color="#6b7280", ha="center")
    fig.suptitle(f"Computational efficiency (mean ± sd over {n_seeds} seeds; single CPU, wall-clock)"); _save(fig, P / "04_computational_efficiency.png")
    # ---- trade-off
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for a, xc, xl in [(ax[0], "Tuning + training time (s)", "tuning + training time (s, log)"), (ax[1], "Inference per record (us, batch avg)", "inference time per record (µs, log)")]:
        for i, k in enumerate(MODELS):
            if k == "K-Means": continue          # chance level: off the zoomed axis, noted in the title
            a.errorbar(e.loc[k, f"{xc} (mean)"], m.loc[k, "Performance score (mean)"], yerr=m.loc[k, "Performance score (std)"], fmt="o", ms=9, color=COL[k], capsize=3)
            a.annotate(k, (e.loc[k, f"{xc} (mean)"], m.loc[k, "Performance score (mean)"]), textcoords="offset points", xytext=[(6, 8), (6, -14), (-30, 12), (6, 8), (6, -14), (6, 8)][i], fontsize=9)
        a.set_xscale("log"); a.set_xlabel(xl); a.set_ylabel("performance score (mean of F1, BalAcc, ROC-AUC, AP)"); a.set_ylim(float((m["Performance score (mean)"].drop("K-Means") - m["Performance score (std)"].fillna(0).drop("K-Means")).min()) - .01, float((m["Performance score (mean)"] + m["Performance score (std)"].fillna(0)).max()) + .01)
    fig.suptitle("Performance vs computational cost (upper-left is better; zoomed, K-Means omitted: score %.2f, chance level)" % m.loc["K-Means", "Performance score (mean)"]); _save(fig, P / "05_performance_vs_cost.png")
    # ---- TSTR
    rm = _ms(real, ["ROC-AUC", "Recall", "Specificity"]); fig, ax = plt.subplots(1, 2, figsize=(13, 4.6)); w = .38; xs = np.arange(len(MODELS))
    ax[0].bar(xs - w / 2, sm["ROC-AUC (mean)"], w, yerr=sm["ROC-AUC (std)"].fillna(0), color="#e45756", label="synthetic test", capsize=2)
    ax[0].bar(xs + w / 2, rm["ROC-AUC (mean)"], w, yerr=rm["ROC-AUC (std)"].fillna(0), color="#4c78a8", label="real test (train-on-synthetic)", capsize=2)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(MODELS, rotation=25, ha="right"); ax[0].set_title("ROC-AUC: synthetic test vs real test"); ax[0].legend(fontsize=8); ax[0].set_ylim(.4, 1.02)
    ax[1].bar(xs - w / 2, rm["Recall (mean)"], w, color="#54a24b", label="recall (positives found)"); ax[1].bar(xs + w / 2, rm["Specificity (mean)"], w, color="#9aa5b1", label="specificity")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(MODELS, rotation=25, ha="right"); ax[1].set_title("Real test set (~10% positives): recall vs specificity at threshold 0.5"); ax[1].legend(fontsize=8)
    _save(fig, P / "06_synthetic_vs_real_test.png")
    # ---- confusion matrices + curves (seed 42)
    pk = pickle.load(open(T / "seed42_synthetic_test_predictions.pkl", "rb")); fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.6))
    for a, k in zip(ax.ravel(), MODELS):
        p = pk[k]; cm = pd.crosstab(p["y_true"], p["y_pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a, xticklabels=["0", "1"], yticklabels=["0", "1"]); a.set_title(k, fontsize=10); a.set_xlabel("Predicted"); a.set_ylabel("True")
    fig.suptitle("Confusion matrices, synthetic test set (seed 42, 500 + 500 records)"); _save(fig, P / "07_confusion_matrices_seed42.png")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for k in MODELS:
        p = pk[k]; f, t, _ = roc_curve(p["y_true"], p["score"]); ax[0].plot(f, t, color=COL[k], label=f"{k} ({auc(f, t):.3f})")
        pr, rc, _ = precision_recall_curve(p["y_true"], p["score"]); ax[1].plot(rc, pr, color=COL[k], label=f"{k} ({average_precision_score(p['y_true'], p['score']):.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].set_title("ROC (seed 42, synthetic test)"); ax[0].legend(fontsize=8, loc="lower right")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall (seed 42, synthetic test)"); ax[1].legend(fontsize=8, loc="lower left"); _save(fig, P / "08_roc_pr_curves_seed42.png")
    print(_md(disp, "{:.4g}")); print(json.dumps(concl, indent=1))
