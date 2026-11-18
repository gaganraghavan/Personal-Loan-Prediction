"""Optimizer/solver and ensemble extensions evaluated on the balanced synthetic dataset.

Run: python -m src.synthetic_extensions
Uses the seed-42 synthetic dataset and a stratified 80/20 split. All selection/CV uses
training data only; the synthetic test split is used only for final reporting.
"""
import ast, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from .evaluation import evaluate, fit_recording
from .preprocessing import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
SYN_DIR = ROOT / 'results' / 'synthetic'
DATA_PATH = SYN_DIR / 'data' / 'synthetic_balanced_seed42.csv'
OUT = SYN_DIR / 'extensions'
SCORING = {'F1': 'f1', 'ROC-AUC': 'roc_auc', 'PR-AUC': 'average_precision', 'Balanced accuracy': 'balanced_accuracy'}


def cv_summary(model, X, y, seed=42):
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always', ConvergenceWarning)
        r = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=1, error_score='raise')
    return {f'CV {k} mean': float(np.mean(r[f'test_{k}'])) for k in SCORING} | {
        f'CV {k} std': float(np.std(r[f'test_{k}'], ddof=1)) for k in SCORING
    } | {'CV convergence warnings': sum(issubclass(w.category, ConvergenceWarning) for w in ws)}


def evaluate_config(label, model, Xtr, ytr, Xte, yte, seed=42):
    row = {'Experiment': label, **cv_summary(model, Xtr, ytr, seed)}
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always', ConvergenceWarning)
        t0 = time.perf_counter(); model.fit(Xtr, ytr); train_s = time.perf_counter() - t0
    metrics, _ = evaluate(model, Xte, yte)
    row.update({f'Test {k}': metrics[k] for k in ['Accuracy','Precision','Recall','F1','Balanced Accuracy','ROC-AUC','PR-AUC']})
    row['Final fit time (s)'] = train_s
    row['Final-fit convergence warnings'] = sum(issubclass(w.category, ConvergenceWarning) for w in ws)
    return row, model


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'{DATA_PATH} missing. Run python -m src.synthetic_experiment first.')
    df = pd.read_csv(DATA_PATH)
    target = 'Personal Loan'
    X = df.drop(columns=[target]).to_numpy(float); y = df[target].to_numpy(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    opt_rows = []
    # Optimizer/solver comparisons: same split and CV folds, train-only CV selection.
    mlp_configs = {
        'MLP Adam (lr=0.001)': dict(solver='adam', learning_rate_init=.001),
        'MLP SGD (lr=0.001)': dict(solver='sgd', learning_rate_init=.001),
        'MLP SGD (lr=0.01)': dict(solver='sgd', learning_rate_init=.01),
        'MLP L-BFGS': dict(solver='lbfgs', learning_rate_init=.001),
    }
    for label, kw in mlp_configs.items():
        est = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', alpha=1e-4, max_iter=100, random_state=42, **kw)
        row, _ = evaluate_config(label, make_pipeline(est, True), Xtr, ytr, Xte, yte)
        row['Model family'] = 'MLP optimizer comparison'; opt_rows.append(row)
    for solver in ['lbfgs','newton-cg','newton-cholesky','liblinear','sag','saga']:
        est = LogisticRegression(solver=solver, C=1.0, max_iter=2000, random_state=42)
        row, _ = evaluate_config(f'Logistic Regression: {solver}', make_pipeline(est, True), Xtr, ytr, Xte, yte)
        row['Model family'] = 'Logistic Regression solver comparison'; opt_rows.append(row)
    for penalty in [None, 'l2', 'l1', 'elasticnet']:
        for eta0 in [.1, 1.0]:
            est = Perceptron(penalty=penalty, alpha=1e-4, eta0=eta0, max_iter=1000, random_state=42)
            row, _ = evaluate_config(f'Perceptron penalty={penalty}, eta0={eta0}', make_pipeline(est, True), Xtr, ytr, Xte, yte)
            row['Model family'] = 'Perceptron configuration comparison'; opt_rows.append(row)
    opt_df = pd.DataFrame(opt_rows)
    opt_df.to_csv(OUT / 'synthetic_optimizer_comparisons.csv', index=False)

    # Ensemble comparison: use best hyperparameters found on synthetic training data in the
    # original synthetic run (seed 42), rather than selecting parameters on the test set.
    params_path = SYN_DIR / 'tables' / 'best_hyperparameters_per_seed.json'
    params_all = json.loads(params_path.read_text())
    def parse_param(v):
        if not isinstance(v, str): return v
        try: return ast.literal_eval(v)
        except (ValueError, SyntaxError): return v
    params = {k: {p: parse_param(v) for p, v in d.items()} for k, d in params_all['42'].items()}
    def component(name, probability=False):
        p = dict(params[name]); p['random_state'] = 42
        if name == 'Logistic Regression': est = LogisticRegression(**p)
        elif name == 'SVM': p['probability'] = probability; est = SVC(**p)
        elif name == 'MLP': est = MLPClassifier(**p)
        else: raise ValueError(name)
        return make_pipeline(est, True)
    estimators = [('lr', component('Logistic Regression')), ('svm', component('SVM', True)), ('mlp', component('MLP'))]
    models = {
        'Soft Voting Ensemble': VotingClassifier(estimators=estimators, voting='soft'),
        'Stacking Ensemble': StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000, random_state=42),
                                                cv=StratifiedKFold(3, shuffle=True, random_state=42),
                                                stack_method='predict_proba', passthrough=False),
    }
    # Include component baselines fitted with the same selected synthetic hyperparameters.
    for name in ['Logistic Regression','SVM','MLP']:
        models[f'{name} (synthetic-tuned baseline)'] = component(name, probability=(name == 'SVM'))
    ens_rows = []
    for label, model in models.items():
        row, _ = evaluate_config(label, model, Xtr, ytr, Xte, yte)
        ens_rows.append(row)
    ens_df = pd.DataFrame(ens_rows)
    ens_df.to_csv(OUT / 'synthetic_ensemble_comparison.csv', index=False)

    # concise markdown summary generated directly from the CSVs
    opt_best = opt_df.sort_values('CV PR-AUC mean', ascending=False).iloc[0]
    ens_best = ens_df[ens_df['Experiment'].str.contains('Ensemble')].sort_values('CV PR-AUC mean', ascending=False).iloc[0]
    report = f'''# Synthetic optimizer and ensemble extension\n\nDataset: balanced synthetic seed 42; {len(df)} rows; class counts: {df[target].value_counts().sort_index().to_dict()}. Split: stratified 80/20. Three-fold stratified CV is performed on the synthetic training split only; the synthetic test split is used for final metrics.\n\n## Optimizer/solver comparison\n\nThe best configuration by training-only CV PR-AUC in this run was **{opt_best['Experiment']}** (CV average precision {opt_best['CV PR-AUC mean']:.4f}). See `synthetic_optimizer_comparisons.csv` for all tested configurations, test metrics, timing, and convergence warnings.\n\n## Ensemble comparison\n\nThe best ensemble by training-only CV PR-AUC was **{ens_best['Experiment']}** (CV average precision {ens_best['CV PR-AUC mean']:.4f}). Soft voting averages the predicted probabilities of Logistic Regression, SVM, and MLP. Stacking trains a logistic-regression meta-model on out-of-fold base-model probabilities. See `synthetic_ensemble_comparison.csv` for the ensemble and component results.\n\n## Interpretation and limitations\n\nThese are extension experiments on the seed-42 synthetic dataset, not five-seed aggregate results. Optimizer/solver settings are compared using the same split and CV folds. Ensemble component hyperparameters come from the synthetic experiment's seed-42 training-only search. Results may vary with the seed and hardware. Labels in the synthetic data are assigned conditionally from the original historical classes; this is a model-comparison experiment, not an independent loan-decision simulator. MLP convergence warnings are recorded rather than silently discarded.\n'''
    (OUT / 'README.md').write_text(report)
    print('Synthetic optimizer comparisons:', OUT / 'synthetic_optimizer_comparisons.csv')
    print('Synthetic ensemble comparison:', OUT / 'synthetic_ensemble_comparison.csv')
    print('Report:', OUT / 'README.md')
    print('\nEnsembles:')
    print(ens_df[['Experiment','CV PR-AUC mean','Test F1','Test ROC-AUC','Final fit time (s)']].to_string(index=False))

if __name__ == '__main__':
    run()
