"""Synthetic balanced Personal-Loan data: class-conditional Gaussian copula + quality validation.

Technique (numpy/scipy only, fully seeded):
  for each class c in {0,1} separately (-> exactly the requested number of rows per class):
    1. marginals  : empirical quantile function of every feature, observed values only (so every generated
                    value is a value that exists in the real data: valid integers/categories/ZIP codes/CCAvg grid).
                    Experience is generated as Age - Gap, where Gap = Age - Experience has its own observed-value
                    marginal (in the real data Gap is always in 24..30).
    2. dependence : Gaussian copula = correlation matrix of the normal scores of the ranks
    3. sampling   : z ~ N(0, R)  ->  u = Phi(z)  ->  x_j = Q_j(u_j)
    4. validity   : Age - Experience and Experience itself are constrained to the real-training ranges (rejection sampling)
Because the model is class-conditional, feature-target relationships are carried by the class-specific
marginals and correlations. The generator is fitted on a real TRAINING split only.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm, rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

TARGET = "Personal Loan"
SMOOTH = ["Age", "Experience", "Income", "Mortgage", "CCAvg"]
NUMERIC_KS = SMOOTH + ["ZIP Code"]


def _decimals(s):
    for k in range(4):
        if np.allclose(s, np.round(s, k)): return k
    return 4


class ClassConditionalCopula:
    """smooth: columns whose marginal is interpolated (others resample observed values only).
    model_gap: model Gap = Age - Experience (discrete, observed values only) instead of Experience."""
    def __init__(self, smooth=(), model_gap=True):
        self.smooth = list(smooth); self.model_gap = model_gap

    def _prep(self, df):
        d = df.copy()
        if self.model_gap: d["Gap"] = d["Age"] - d["Experience"]; d = d.drop(columns="Experience")
        return d

    def fit(self, df):
        self.out_cols = [c for c in df.columns if c != TARGET]; self.dtypes = df.dtypes
        gap = df["Age"] - df["Experience"]; self.gap = (float(gap.min()), float(gap.max()))
        self.experience = (float(df["Experience"].min()), float(df["Experience"].max()))
        d = self._prep(df); self.cols = [c for c in d.columns if c != TARGET]
        self.dec = {c: _decimals(d[c].to_numpy(float)) for c in self.smooth if c in self.cols}; self.m = {}
        for c in (0, 1):
            X = d.loc[d[TARGET] == c, self.cols]; n = len(X)
            Z = np.column_stack([norm.ppf(rankdata(X[k]) / (n + 1)) for k in self.cols])
            with np.errstate(invalid="ignore"): R = np.nan_to_num(np.corrcoef(Z, rowvar=False))
            np.fill_diagonal(R, 1.0); w, V = np.linalg.eigh(R); R = (V * np.clip(w, 1e-6, None)) @ V.T
            dg = np.sqrt(np.diag(R)); R = R / np.outer(dg, dg)
            self.m[c] = {"sorted": {k: np.sort(X[k].to_numpy(float)) for k in self.cols}, "L": np.linalg.cholesky(R), "n": n}
        return self

    def _draw(self, c, m, rng):
        mod = self.m[c]; u = norm.cdf(rng.standard_normal((m, len(self.cols))) @ mod["L"].T); out = {}
        for j, k in enumerate(self.cols):
            s = mod["sorted"][k]
            out[k] = (np.round(np.quantile(s, u[:, j], method="linear"), self.dec[k]) if k in self.dec
                      else np.quantile(s, u[:, j], method="inverted_cdf"))
        out = pd.DataFrame(out)
        if self.model_gap: out["Experience"] = out["Age"] - out["Gap"]; out = out.drop(columns="Gap")
        return out[self.out_cols]

    def generate(self, n_per_class, seed):
        rng = np.random.default_rng(seed); parts = []; self.acceptance = {}
        for c in (0, 1):
            got, need, drawn, acc = [], n_per_class, 0, 0
            while need > 0:
                d = self._draw(c, int(need * 1.5) + 10, rng); g = d["Age"] - d["Experience"]
                ok = d[(g >= self.gap[0]) & (g <= self.gap[1]) &
                       (d["Experience"] >= self.experience[0]) &
                       (d["Experience"] <= self.experience[1])]; drawn += len(d); acc += len(ok)
                got.append(ok.iloc[:need]); need -= len(got[-1])
            part = pd.concat(got); part[TARGET] = c; parts.append(part); self.acceptance[c] = acc / drawn
        out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        return out.astype(self.dtypes.to_dict())


# ------------------------------------------------------------------ quality validation
def _tvd(a, b):
    p = pd.concat([a.value_counts(normalize=True), b.value_counts(normalize=True)], axis=1).fillna(0)
    return 0.5 * float(np.abs(p.iloc[:, 0] - p.iloc[:, 1]).sum())


def _dist(a, b, k): return ks_2samp(a, b).statistic if k in NUMERIC_KS else _tvd(a, b)


def _cohen_d(df):
    a, b = df[df[TARGET] == 1], df[df[TARGET] == 0]; cols = [c for c in df.columns if c != TARGET]
    sd = np.sqrt((a[cols].var() + b[cols].var()) / 2).replace(0, np.nan)
    return ((a[cols].mean() - b[cols].mean()) / sd).fillna(0)


def _detect(a, b, seed, cap=1000):
    """Real-vs-synthetic detectability on FEATURES ONLY with equal group sizes (a random forest, 5-fold CV ROC-AUC;
    0.5 = indistinguishable). Must be called on rows of ONE class: the target column and the class mix are excluded,
    because real data is ~10% positive and synthetic data 50% positive by design (the label alone gives AUC ~0.70)."""
    n = min(len(a), len(b), cap); A = a.sample(n, random_state=seed); B = b.sample(n, random_state=seed)
    X = pd.concat([A, B]).to_numpy(float); y = np.r_[np.zeros(n), np.ones(n)]
    rf = RandomForestClassifier(100, random_state=seed, n_jobs=1)
    return float(cross_val_score(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc").mean())


def _dups(a, b):
    return float(len(a.merge(b.drop_duplicates(), how="inner")) / len(a))


def quality_report(real_train, real_test, syn, seed):
    """Compare synthetic data with the real TRAIN split (generator source) and the real TEST split (held out).
    Real-train vs real-test is the 'sampling noise floor' reference."""
    cols = [c for c in real_train.columns if c != TARGET]; marg, res = [], {}
    for c in (0, 1):
        r, t, s = (d[d[TARGET] == c] for d in (real_train, real_test, syn))
        for k in cols:
            marg.append({"Class": c, "Feature": k, "Metric": "KS" if k in NUMERIC_KS else "TVD",
                         "Synthetic vs real-train": _dist(s[k], r[k], k), "Real-test vs real-train (noise floor)": _dist(t[k], r[k], k)})
    marg = pd.DataFrame(marg)
    corr = {}
    for c in (0, 1):
        R, T, S = (d[d[TARGET] == c][cols].corr(method="spearman").fillna(0).to_numpy() for d in (real_train, real_test, syn))
        off = ~np.eye(len(cols), dtype=bool); corr[c] = (np.abs(S - R)[off].mean(), np.abs(T - R)[off].mean())
    dr, ds = _cohen_d(real_train), _cohen_d(syn); eff = pd.DataFrame({"Real-train Cohen d": dr, "Synthetic Cohen d": ds, "Abs diff": (dr - ds).abs()})
    sc = MinMaxScaler().fit(real_train[cols]); nn = NearestNeighbors(n_neighbors=1).fit(sc.transform(real_train[cols]))
    det, dcr = {}, {}
    for c in (0, 1):   # per class, features only, same-size groups (see _detect); distances to the nearest SAME-CLASS real-train record
        r_, t_, s_ = (d[d[TARGET] == c][cols] for d in (real_train, real_test, syn)); n = min(len(r_), len(t_)); nn = NearestNeighbors(n_neighbors=1).fit(sc.transform(r_))
        det[f"Detectability AUC class{c}: synthetic vs real-train (matched n)"] = _detect(r_, s_, seed, cap=n)
        det[f"Detectability AUC class{c}: real-test vs real-train (matched n, noise floor)"] = _detect(r_, t_, seed, cap=n)
        det[f"Detectability AUC class{c}: synthetic vs real-train (larger n, up to 1000)"] = _detect(r_, s_, seed, cap=1000)
        for lab, d in (("synthetic", s_), ("real-test", t_)):
            sample_n = min(n, len(d))
            dcr[f"Median nearest same-class real-train distance class{c}: {lab}"] = float(np.median(nn.kneighbors(sc.transform(d.sample(sample_n, random_state=seed)))[0]))
    gap = syn["Age"] - syn["Experience"]
    res = {"Seed": seed, "Rows": len(syn), "Class 0 rows": int((syn[TARGET] == 0).sum()), "Class 1 rows": int((syn[TARGET] == 1).sum()),
           "Violations: Age-Experience outside real range": float(((gap < real_train["Age"].sub(real_train["Experience"]).min()) | (gap > real_train["Age"].sub(real_train["Experience"]).max())).mean()),
           "Violations: values outside observed range": float(((syn[cols] < real_train[cols].min()) | (syn[cols] > real_train[cols].max())).any(axis=1).mean()),
           "Mean KS/TVD synthetic vs real-train": float(marg["Synthetic vs real-train"].mean()),
           "Mean KS/TVD real-test vs real-train (floor)": float(marg["Real-test vs real-train (noise floor)"].mean()),
           "Spearman mean abs diff class0 (syn vs real)": corr[0][0], "Spearman mean abs diff class1 (syn vs real)": corr[1][0],
           "Spearman mean abs diff class0 (floor)": corr[0][1], "Spearman mean abs diff class1 (floor)": corr[1][1],
           "Cohen d mean abs diff": float(eff["Abs diff"].mean()), "Cohen d correlation (real vs syn)": float(np.corrcoef(dr, ds)[0, 1]),
           "Exact duplicate rate synthetic -> real-train": _dups(syn, real_train), "Exact duplicate rate real-test -> real-train (reference)": _dups(real_test, real_train)}
    res.update(det); res.update(dcr)
    return res, marg, eff
