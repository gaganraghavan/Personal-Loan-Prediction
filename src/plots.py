"""Plotting helpers (matplotlib/seaborn). Only plots that answer a stated question."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

sns.set_theme(style="whitegrid", context="notebook")
VERSIONS = ["Baseline (paper-faithful)", "Baseline (all scaled)", "Tuned"]
VCOL = dict(zip(VERSIONS, ["#9aa5b1", "#4c78a8", "#e45756"]))


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def class_distribution(y, ytr, yte, path):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    d = pd.DataFrame({"Full (n=%d)" % len(y): y.value_counts(), "Train (n=%d)" % len(ytr): ytr.value_counts(),
                      "Test (n=%d)" % len(yte): yte.value_counts()}).T.rename(columns={0: "Declined (0)", 1: "Accepted (1)"})
    d.plot.bar(ax=ax, color=["#9aa5b1", "#e45756"], rot=0)
    for c in ax.containers: ax.bar_label(c, fontsize=8)
    pct = 100 * y.mean(); ax.set_title(f"Class distribution (positive rate = {pct:.1f}%)"); ax.set_ylabel("Customers")
    _save(fig, path)


def confusion_grid(preds, models, path):
    versions = ["Baseline (paper-faithful)", "Tuned"]
    fig, axes = plt.subplots(2, len(models), figsize=(2.7 * len(models), 5.6))
    for i, v in enumerate(versions):
        for j, m in enumerate(models):
            ax = axes[i, j]; p = preds.get((m, v))
            if p is None:
                ax.axis("off"); ax.text(.5, .5, "not tuned\n(see README)", ha="center", va="center", fontsize=8); continue
            cm = pd.crosstab(p["y_true"], p["y_pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["0", "1"], yticklabels=["0", "1"])
            ax.set_title(f"{m}\n{'Baseline' if i == 0 else 'Tuned'}", fontsize=9)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True" if j == 0 else "")
    _save(fig, path)


def curves(preds, models, kind, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, v in zip(axes, ["Baseline (paper-faithful)", "Tuned"]):
        for m in models:
            p = preds.get((m, v))
            if p is None: continue
            if kind == "roc":
                f, t, _ = roc_curve(p["y_true"], p["score"]); ax.plot(f, t, label=f"{m} (AUC={auc(f, t):.3f})")
            else:
                pr, rc, _ = precision_recall_curve(p["y_true"], p["score"])
                ax.plot(rc, pr, label=f"{m} (AP={average_precision_score(p['y_true'], p['score']):.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlabel("False positive rate")
        else:
            ax.axhline(p["y_true"].mean(), color="k", ls=":", lw=1); ax.set_xlabel("Recall")
        ax.set_title(v); ax.legend(fontsize=8, loc="lower right" if kind == "roc" else "lower left")
    axes[0].set_ylabel("True positive rate" if kind == "roc" else "Precision")
    _save(fig, path)


def baseline_vs_tuned(comp, path, metrics=("ROC-AUC", "PR-AUC", "F1", "Recall")):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, m in zip(axes.ravel(), metrics):
        sns.barplot(data=comp, x="Model", y=m, hue="Version", hue_order=VERSIONS, palette=VCOL, ax=ax)
        ax.set_title(m); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
        if ax is not axes[0, 0]: ax.get_legend().remove()
    h, l = axes[0, 0].get_legend_handles_labels(); axes[0, 0].get_legend().remove()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Test-set performance: paper baseline vs scaled baseline vs tuned", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def optimizer_plot(mlp, lr, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, df, title in [(axes[0], mlp, "MLP optimizer (5-fold CV on train)"), (axes[1], lr, "LogReg solver (5-fold CV on train)")]:
        ax.bar(df["Optimizer"], df["CV average_precision"], color="#4c78a8")
        ax.set_ylim(max(0, df["CV average_precision"].min() - 0.05), min(1.0, df["CV average_precision"].max() + 0.03))
        ax.set_title(title + ("  [truncated y-axis]" if "LogReg" in title else "")); ax.set_ylabel("CV PR-AUC"); ax.tick_params(axis="x", rotation=30); [t.set_ha("right") for t in ax.get_xticklabels()]
    ax = axes[2]
    ax.bar(mlp["Optimizer"], mlp["Test PR-AUC (mean)"], yerr=mlp["Test PR-AUC (std over init seeds)"], color="#e45756", capsize=4)
    ax.set_title("MLP: test PR-AUC (mean ± sd, 5 init seeds)", fontsize=10); ax.tick_params(axis="x", rotation=30); [t.set_ha("right") for t in ax.get_xticklabels()]
    ax.set_ylim(max(0, (mlp["Test PR-AUC (mean)"].min() - 0.08)), 1.0)
    _save(fig, path)


def sensitivity_plot(cv_tables, sens, path):
    models = list(cv_tables)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8)); axes = axes.ravel()
    for ax, m in zip(axes, models):
        s = sens[sens["Model"] == m].sort_values("Marginal range of mean CV PR-AUC", ascending=False)
        top = s.iloc[0]["Hyperparameter"]; df = cv_tables[m].copy()
        df["val"] = df["param_clf__" + top].astype(str)
        def _k(v):
            try: return (0, float(v), v)
            except ValueError: return (1, 0.0, v)
        order = sorted(df["val"].unique(), key=_k)
        sns.boxplot(data=df, x="val", y="mean_test_average_precision", order=order, ax=ax, color="#9ecae1")
        sns.stripplot(data=df, x="val", y="mean_test_average_precision", order=order, ax=ax, color="k", size=2.5, alpha=.5)
        ax.set_title(f"{m}: most sensitive = {top}"); ax.set_xlabel(top); ax.set_ylabel("CV PR-AUC (per config)")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    for ax in axes[len(models):]: ax.axis("off")
    _save(fig, path)


# ---------------------------------------------------------------------------------------------
# Ensemble-extension plots (appended; existing plot functions above are unchanged)
# ---------------------------------------------------------------------------------------------
ENS_COLORS = {"Soft Voting Ensemble": "#e45756", "Stacking Ensemble": "#f58518"}
IND_COLORS = {"Logistic Regression (tuned)": "#9aa5b1", "SVM (tuned)": "#4c78a8", "MLP (tuned)": "#54a24b"}


def _ens_color(label): return ENS_COLORS.get(label) or IND_COLORS.get(label, "#4c78a8")


def ensemble_metrics(cmp_df, best_label, path):
    """ROC-AUC, PR-AUC, F1, Recall: tuned individual models vs the two ensemble extensions (test set)."""
    labels = list(cmp_df["Model"]); ticks = [l + (" *" if l == best_label else "") for l in labels]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, m in zip(axes.ravel(), ["ROC-AUC", "PR-AUC", "F1", "Recall"]):
        v = cmp_df[m].values; ax.bar(range(len(v)), v, color=[_ens_color(l) for l in labels])
        for i, x in enumerate(v): ax.text(i, x, f"{x:.3f}", ha="center", va="bottom", fontsize=8)
        lo = 0.9 if m == "ROC-AUC" else 0.0; ax.set_ylim(lo, 1.004 if m == "ROC-AUC" else 1.05)
        ax.set_title(m + ("  [y-axis starts at 0.9]" if m == "ROC-AUC" else "")); ax.set_xticks(range(len(v)))
        ax.set_xticklabels(ticks, rotation=25, ha="right", fontsize=8)
    fig.suptitle("Ensemble extension vs tuned individual models (test set; * = best individual by train-only CV PR-AUC)", y=1.0)
    _save(fig, path)


def ensemble_curves(preds, path):
    """ROC and Precision-Recall curves of the tuned LR/SVM/MLP and both ensembles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for lab, p in preds.items():
        kw = dict(color=_ens_color(lab), lw=2.2 if lab in ENS_COLORS else 1.4, ls="-" if lab in ENS_COLORS else "--")
        f, t, _ = roc_curve(p["y_true"], p["score"]); axes[0].plot(f, t, label=f"{lab} (AUC={auc(f, t):.3f})", **kw)
        pr, rc, _ = precision_recall_curve(p["y_true"], p["score"]); axes[1].plot(rc, pr, label=f"{lab} (AP={average_precision_score(p['y_true'], p['score']):.3f})", **kw)
    axes[0].plot([0, 1], [0, 1], "k:", lw=1); axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate"); axes[0].set_title("ROC")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-Recall")
    for ax in axes: ax.legend(fontsize=7.5)
    axes[0].set_xlim(0, 0.3); axes[0].set_ylim(0.5, 1.01); axes[0].set_title("ROC (zoom: FPR 0-0.3, TPR 0.5-1)")
    _save(fig, path)


def ensemble_confusions(preds, path):
    fig, axes = plt.subplots(1, len(preds), figsize=(2.9 * len(preds), 3.2))
    for ax, (lab, p) in zip(axes, preds.items()):
        cm = pd.crosstab(p["y_true"], p["y_pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["0", "1"], yticklabels=["0", "1"])
        ax.set_title(lab, fontsize=8); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    _save(fig, path)
