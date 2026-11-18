"""Extra check of a paper claim: MLP with vs without MinMaxScaler
(everything else identical; 10 init seeds, same 80/20 split)."""
from pathlib import Path
import pandas as pd
from sklearn.neural_network import MLPClassifier
from .evaluation import evaluate, fit_recording
from .preprocessing import SEED, load_data, make_pipeline, split_data

if __name__ == "__main__":
    X, y = load_data(); Xtr, Xte, ytr, yte = split_data(X, y, seed=SEED); rows = []
    for scaled in (False, True):
        for s in range(10):
            m = make_pipeline(MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam",
                                            alpha=1e-4, learning_rate="constant", random_state=s), scaled)
            _, conv = fit_recording(m, Xtr, ytr); met, _ = evaluate(m, Xte, yte)
            rows.append({"MinMaxScaler": scaled, "Init seed": s, "ConvergenceWarning": conv, **met})
    df = pd.DataFrame(rows); out = Path(__file__).resolve().parents[1] / "results"
    df.to_csv(out / "mlp_scaling_check_runs.csv", index=False)
    s = df.groupby("MinMaxScaler")[["Accuracy", "Recall", "F1", "ROC-AUC", "PR-AUC", "FP", "FN", "ConvergenceWarning"]].agg(["mean", "std"])
    s.to_csv(out / "mlp_scaling_check_summary.csv"); print(s.round(3).T.to_string())
