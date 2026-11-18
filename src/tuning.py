"""Phase 5: hyperparameter tuning. Only the TRAINING split is ever passed in here.

* 5-fold StratifiedKFold CV, scalers inside pipelines (re-fitted per fold).
* Refit criterion: PR-AUC (average precision) -- chosen BEFORE seeing results because
  accuracy is uninformative at ~10% prevalence and PR-AUC is threshold-free and
  sensitive to the minority class. Other metrics are recorded for every candidate.
* All tuned pipelines use MinMaxScaler (extension; compare against 'Baseline (all scaled)').
"""
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from .preprocessing import make_pipeline

SCORING = {"average_precision": "average_precision", "roc_auc": "roc_auc", "f1": "f1",
           "recall": "recall", "precision": "precision",
           "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy"}
REFIT = "average_precision"
N_SPLITS = 5
MLP_N_ITER = 30


def search_spaces(seed):
    P = "clf__"
    def pre(d): return {P + k: v for k, v in d.items()}
    return {
        "Perceptron": ("grid", Perceptron(random_state=seed), pre({
            "penalty": [None, "l2", "l1", "elasticnet"], "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "max_iter": [100, 1000], "eta0": [0.1, 1.0]})),
        "Logistic Regression": ("grid", LogisticRegression(max_iter=2000, random_state=seed), [
            # l1_ratio: 0 -> pure L2, 1 -> pure L1 (sklearn>=1.8 API; replaces 'penalty')
            pre({"solver": ["lbfgs", "newton-cg", "newton-cholesky"], "l1_ratio": [0.0],
                 "C": [0.01, 0.1, 1, 10, 100, 1000]}),
            pre({"solver": ["liblinear"], "l1_ratio": [0.0, 1.0], "C": [0.01, 0.1, 1, 10, 100, 1000]}),
            pre({"solver": ["saga"], "l1_ratio": [0.0, 0.5, 1.0], "C": [0.01, 0.1, 1, 10, 100, 1000]}),
        ]),
        "KNeighbors": ("grid", KNeighborsClassifier(), pre({
            "n_neighbors": [1, 3, 5, 7, 11, 15, 21, 31], "weights": ["uniform", "distance"], "p": [1, 2]})),
        "SVM": ("grid", SVC(), [
            pre({"kernel": ["linear"], "C": [0.1, 1, 10, 100]}),
            pre({"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]}),
            pre({"kernel": ["poly"], "C": [0.1, 1, 10], "degree": [2, 3], "gamma": ["scale"]}),
        ]),
        "MLP": ("random", MLPClassifier(random_state=seed), pre({
            "hidden_layer_sizes": [(50,), (100,), (100, 50), (64, 32), (128, 64, 32)],
            "activation": ["relu", "tanh"], "solver": ["adam", "sgd", "lbfgs"],
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2], "learning_rate": ["constant", "adaptive"],
            "learning_rate_init": [1e-3, 1e-2, 1e-1], "batch_size": [64, 200],
            "max_iter": [300, 500]})),
    }


def run_search(name, kind, estimator, space, Xtr, ytr, seed):
    pipe = make_pipeline(estimator, scaled=True)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    kw = dict(scoring=SCORING, refit=REFIT, cv=cv, n_jobs=1, return_train_score=False, error_score=np.nan)
    gs = (GridSearchCV(pipe, space, **kw) if kind == "grid"
          else RandomizedSearchCV(pipe, space, n_iter=MLP_N_ITER, random_state=seed, **kw))
    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gs.fit(Xtr, ytr)
    cvdf = pd.DataFrame(gs.cv_results_)
    return gs, cvdf, time.time() - t
