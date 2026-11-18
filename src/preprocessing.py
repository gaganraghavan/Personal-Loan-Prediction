"""Data loading, splitting and (leak-free) scaling.

Scalers are ALWAYS placed inside an sklearn Pipeline, so they are fitted on the
training portion only (and, inside cross-validation, on each training fold only).
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

SEED = 42
TARGET = "Personal Loan"
ID_COLUMNS = ["ID"]  # the paper removes the 1..5000 index column
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Personal_Loan.csv"


def load_data(path=DATA_PATH):
    """Return (X, y). 12 features remain after dropping ID and the target,
    matching the paper's '12 features and a [0,1] label'.
    ZIP Code, Education (1/2/3) and Experience (incl. 52 negative values) are kept
    as given: the paper describes no extra preprocessing."""
    df = pd.read_csv(path)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET] + ID_COLUMNS)
    return X, y


def split_data(X, y, seed=SEED, test_size=0.2, stratify=True):
    """80/20 split. The paper gives the ratio only (no seed, no stratification
    stated). We use a fixed seed and stratify (documented deviation/choice)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if stratify else None
    )


def make_pipeline(estimator, scaled):
    steps = ([("scaler", MinMaxScaler())] if scaled else []) + [("clf", estimator)]
    return Pipeline(steps)
