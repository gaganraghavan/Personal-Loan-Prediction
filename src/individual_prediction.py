"""Terminal prediction for one customer using the project's tuned real-data models.

The models are fit on the fixed real training split (seed 42), with the tuned
hyperparameters saved by src.experiments. The real test split is held out for
reporting evaluation metrics and is never used to fit these models.

Run from the project root:
    python -m src.individual_prediction
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .preprocessing import SEED, load_data, split_data, make_pipeline

ROOT = Path(__file__).resolve().parents[1]
HP_PATH = ROOT / "results" / "best_hyperparameters.json"
MODEL_DIR = ROOT / "models"
METRICS_PATH = ROOT / "results" / "individual_prediction_holdout_metrics.csv"
FEATURES = ["Age", "Experience", "Income", "ZIP Code", "Family", "CCAvg",
            "Education", "Mortgage", "Securities Account", "CD Account",
            "Online", "CreditCard"]
DISPLAY_NAMES = {0: "Not accepted (0)", 1: "Accepted (1)"}


def _parse_value(value):
    """Decode JSON values saved as strings by the project's tuning script."""
    import ast
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def make_tuned_models(hyperparameters):
    lr_hp = {k: _parse_value(v) for k, v in hyperparameters["Logistic Regression"].items()}
    svm_hp = {k: _parse_value(v) for k, v in hyperparameters["SVM"].items()}
    mlp_hp = {k: _parse_value(v) for k, v in hyperparameters["MLP"].items()}
    # Keep the same tuned model families and MinMaxScaler pipeline used in tuning.
    return {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=2000, random_state=SEED, **lr_hp), scaled=True),
        "SVM": make_pipeline(SVC(**svm_hp), scaled=True),
        "MLP": make_pipeline(MLPClassifier(random_state=SEED, **mlp_hp), scaled=True),
    }


def evaluate_holdout(models, X_test, y_test):
    rows = []
    for name, model in models.items():
        pred = model.predict(X_test)
        # Use ranking scores for ROC-AUC / AP; never label these as probabilities.
        if hasattr(model, "decision_function"):
            score = model.decision_function(X_test)
        elif hasattr(model, "predict_proba"):
            score = model.predict_proba(X_test)[:, 1]
        else:
            score = pred
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, score),
            "Average Precision": average_precision_score(y_test, score),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "Test records": int(len(y_test)), "Positive records": int(np.sum(y_test)),
        })
    return pd.DataFrame(rows)


def read_customer():
    print("\nEnter the customer's 12 feature values. Use the same units/coding as the dataset.")
    print("ZIP Code is numeric in this dataset; Education is coded 1, 2, or 3; binary fields use 0/1.\n")
    values = {}
    specifications = [
        ("Age", int, "Age in years"),
        ("Experience", int, "Years of experience (dataset coding)"),
        ("Income", float, "Annual income (dataset units, $000s)"),
        ("ZIP Code", int, "ZIP code"),
        ("Family", int, "Family size"),
        ("CCAvg", float, "Average monthly credit-card spend (dataset units, $000s)"),
        ("Education", int, "Education level (1, 2, or 3)"),
        ("Mortgage", float, "Mortgage value (dataset units, $000s)"),
        ("Securities Account", int, "Securities account? (0=no, 1=yes)"),
        ("CD Account", int, "CD account? (0=no, 1=yes)"),
        ("Online", int, "Uses online banking? (0=no, 1=yes)"),
        ("CreditCard", int, "Has bank credit card? (0=no, 1=yes)"),
    ]
    for feature, converter, description in specifications:
        while True:
            raw = input(f"{feature} — {description}: ").strip()
            try:
                val = converter(raw)
                if feature in {"Securities Account", "CD Account", "Online", "CreditCard"} and val not in (0, 1):
                    raise ValueError("enter 0 or 1")
                if feature == "Education" and val not in (1, 2, 3):
                    raise ValueError("enter 1, 2, or 3")
                if feature in {"Age", "Income", "Family", "CCAvg", "Mortgage"} and val < 0:
                    raise ValueError("value cannot be negative")
                values[feature] = val
                break
            except ValueError as exc:
                print(f"Invalid value ({exc}). Please try again.")
    return pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)


def main():
    if not HP_PATH.exists():
        raise FileNotFoundError(f"Missing tuned hyperparameters: {HP_PATH}. Run `python -m src.experiments` first.")
    hyperparameters = json.loads(HP_PATH.read_text(encoding="utf-8"))
    X, y = load_data()
    # Match the project's documented fixed, stratified real-data split.
    X_train, X_test, y_train, y_test = split_data(X, y, seed=SEED)
    if list(X.columns) != FEATURES:
        raise ValueError(f"Feature order mismatch. Dataset has {list(X.columns)}; expected {FEATURES}.")

    models = make_tuned_models(hyperparameters)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    for name, model in models.items():
        print(f"Training {name} on {len(X_train)} real training records...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train, y_train)
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODEL_DIR / f"individual_prediction_{safe_name}.joblib")

    metrics = evaluate_holdout(models, X_test, y_test)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    print("\nHELD-OUT REAL TEST PERFORMANCE")
    print(f"Test split: {len(y_test)} customers; accepted cases: {int(y_test.sum())}; declined cases: {int((y_test == 0).sum())}")
    show_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "Balanced Accuracy", "ROC-AUC", "Average Precision"]
    print(metrics[show_cols].round(3).to_string(index=False))
    print(f"\nFull metrics including confusion-matrix counts saved to: {METRICS_PATH.relative_to(ROOT)}")
    print(f"Trained pipelines saved in: {MODEL_DIR.relative_to(ROOT)}/")
    print("\nNote: these are held-out real-test metrics. No probability is presented as a calibrated real-world acceptance probability.")

    while True:
        customer = read_customer()
        print("\nINDIVIDUAL CUSTOMER PREDICTIONS")
        for name, model in models.items():
            prediction = int(model.predict(customer)[0])
            print(f"{name:22s}: {DISPLAY_NAMES[prediction]}")
        print("\nThese are model classifications, not a loan approval decision or a calibrated probability.")
        again = input("\nPredict another customer? [y/N]: ").strip().lower()
        if again not in {"y", "yes"}:
            break


if __name__ == "__main__":
    main()
