# Replication + extension: "Predicting Acceptance of Personal Loans Using ML Algorithms" (Rossman & Casey)

## Project overview

This repository reproduces and extends a personal-loan acceptance classification study using the 5,000-customer Kaggle Personal Loan Modelling dataset. The target is `Personal Loan` (0 = did not accept; 1 = accepted). The included data copy came from a public GitHub mirror rather than directly from Kaggle; see the data note below.

Six model families are implemented: Perceptron, Logistic Regression, K-Means, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP). The work includes paper-style baselines, hyperparameter tuning, five-seed balanced synthetic-data experiments, optimizer/solver and Perceptron comparisons, soft voting and stacking, and an individual-customer prediction CLI.

Preprocessing drops the customer ID and retains the 12 predictors. Data splits are stratified; scaling is applied in model pipelines so it is fit only on training data and within each CV fold. The minority positive class is about 9.6% in the original data, so evaluation reports precision, recall, F1, balanced accuracy, ROC-AUC, PR-AUC (average precision), and confusion-matrix counts alongside accuracy. Search and headline results use the fixed seed-42 real-data split unless a section explicitly says otherwise.

The balanced synthetic study is a separate five-seed experiment (seeds 42–46). Its class-conditional generator is fit on each real training split and its generated data is validated for balance, ranges, duplicates, and distribution/correlation differences. The optimizer and ensemble experiment in `src/synthetic_extensions.py` is a separate seed-42 extension. The individual-prediction holdout metrics come from a separate real-data seed-42 evaluation. These results are reported separately and must not be combined or described as five-seed averages.

The paper baseline and tuning results are summarized in `SUMMARY.md`; the synthetic procedure and its five-seed results are in `SYNTHETIC_EXPERIMENT.md`; implementation checks are in `FINAL_VERIFICATION.md` and `results/verification/VERIFICATION_REPORT.md`. The paper replication is imperfect: discrepancies, single-split uncertainty, class imbalance, and limits of the synthetic generator are documented in those reports. This project describes patterns in historical data and is not a loan approval system.

Replicates the paper's setup (Kaggle Personal Loan Modelling, 5,000 customers, 80/20 split, MinMaxScaler for the MLP, ROC/AUC),
then extends it with CV hyperparameter tuning, optimizer/solver comparison and imbalance-aware metrics.
**Extensions are labelled as such; the paper-replication baselines are never modified.**

## Run
```bash
pip install -r requirements.txt
python -m src.experiments          # ~11 min on 1 CPU; writes everything under results/
python -m src.mlp_scaling_check    # optional: MLP with/without MinMaxScaler over 10 init seeds
python -m src.synthetic_experiment # separate balanced synthetic study, five seeds
python -m src.synthetic_extensions # optional seed-42 synthetic optimizer/ensemble extension
python -m src.individual_prediction # fit/evaluate models, then prompt for one customer
```

The main experiment and individual-prediction commands write result files and trained pipelines. Existing outputs are included in this repository; rerunning these commands may replace them. The synthetic extensions write their own files under `results/synthetic/extensions/`.
Data: `data/Personal_Loan.csv` (5,000 rows, 480 positives). Our copy came from a public GitHub mirror of the Kaggle file, not from Kaggle directly; replace it with your own download if desired (same columns). `data/synthetic_balanced_personal_loan.csv` is a convenience copy of the generated balanced dataset for seed 42 (5,000 newly generated rows, 2,500 per loan-decision class).

## Layout
`src/preprocessing.py` load/split/scaler-in-pipeline · `models.py` baselines + `KMeansClassifier` · `baseline.py` baselines + 10-split variability ·
`tuning.py` search spaces and CV · `optimizers.py` solver comparisons · `evaluation.py` metrics · `plots.py` · `experiments.py` orchestrator.

Top-level folders: `src/` contains implementation; `data/` contains the original and seed-42 convenience datasets; `models/` contains the fitted individual-prediction pipelines; `notebooks/` contains the analysis notebook; `results/` contains experiment tables, model predictions, verification records, and plots. `Personal_Loan_Prediction_Report.pdf` is the final report.

## Design decisions (not specified by the paper)
* Split: 80/20, `random_state=42`, **stratified** (paper: ratio only).
* Dropped `ID`; 12 features remain, as in the paper. ZIP Code, Education and Experience (52 negative values) left as given.
* Scaling is inside a `Pipeline`, so it is fitted on training data (and on each training fold in CV) only.
* **Two baseline variants**: *paper-faithful* (only MLP scaled, as the paper describes) and *all scaled* (control to separate scaling from tuning).
* **K-Means vs KNeighbors**: the paper's methodology says K-Means, its ROC legend says `KNeighborsClassifier`. Both are run and reported separately. K-Means uses 2 clusters; the cluster with the higher training positive rate is called class 1 (the paper does not say how clusters map to labels).
* MLP "100 hidden layers": **our interpretation, not a fact stated by the paper** -> `hidden_layer_sizes=(100,)` (one layer of 100 units; relu, adam, constant LR, alpha=1e-4; everything else sklearn defaults incl. max_iter=200, which raises a ConvergenceWarning).
* K-Means deviation: we use k-means++ with n_init=10; the paper describes random-data-point initialisation. The literal version was also tested (see `results/verification/`) and gives the same result.
* Tuning: 5-fold StratifiedKFold on the **training split only**; refit criterion = PR-AUC (average precision). Test set is touched once, after tuning. All tuned pipelines use MinMaxScaler. K-Means is not tuned.
* Perceptron has no solver choice; only regularisation and `eta0` are compared. Class balancing (SMOTE etc.) was deliberately **not** run.

## Outputs (`results/`)
`baseline_results.csv`, `tuned_results.csv`, `model_comparison.csv`, `paper_vs_replication.csv`, `best_hyperparameters.json`,
`final_model_hyperparameters_all.json` (every parameter of every final model), `optimizer_comparison_{mlp,logreg,perceptron}.csv`,
`baseline_seed_variability_summary.csv`, `mlp_init_seed_summary.csv`, `mlp_scaling_check_summary.csv`, `hyperparameter_sensitivity.csv`, `cv_results/`, `plots/`.

## Plots
01 class distribution · 02 confusion matrices (baseline/tuned) · 03 ROC · 04 Precision-Recall · 05 baseline vs tuned (ROC-AUC, PR-AUC, F1, recall) · 06 optimizer comparison · 07 hyperparameter sensitivity.

## Caveats
Single seed-42 split for headline numbers (96 test positives, so recall has SE ~0.03); **limited MLP search: 30 random candidates out of 2,880 possible combinations (~1%), not an exhaustive optimisation**; default 0.5 threshold; see `SUMMARY.md`.

## Verification
`python -m src.verification {quick|lr|km|opt}` re-checks the methodology; outputs and the verdict list are in `results/verification/` (start with `VERIFICATION_REPORT.md`).

## Ensemble Learning Extension (NEW - not part of the original paper)
**Why:** requested as an additional research extension: *does combining the strongest individually tuned models improve loan-acceptance prediction compared with the individual models?* The paper uses no ensembles, so everything here is labelled `Extension - Soft Voting` / `Extension - Stacking` and is kept apart from the paper replication. The original results (`model_comparison.csv`, `paper_vs_replication.csv`, ...) are untouched.

**What was combined:** Logistic Regression + SVM + MLP, each with the hyperparameters already selected by the existing tuning (read from `results/final_model_hyperparameters_all.json`; **no new search**). Each component is `MinMaxScaler -> estimator` in a Pipeline. The SVM is rebuilt with `probability=True` (Platt scaling; fixed `random_state=42`) only for the ensembles; the existing SVM is unchanged and gives identical hard labels. K-Means (no probabilities), Perceptron (no probabilities) and KNN (weak recall in the tuned version, not among the strongest models) were deliberately not used.
* **Soft voting** - `VotingClassifier(voting="soft")`, equal weights: the mean of the three predicted probabilities, class 1 if > 0.5.
* **Stacking** - `StackingClassifier`; Logistic Regression meta-model (default settings, not tuned) fitted on out-of-fold probabilities from `StratifiedKFold(5, shuffle=True, random_state=42)` on the **training split only**; base models are then refit on the full training split.

**Evaluation / leakage:** same split (seed 42), same 0.5 threshold and the same `evaluation.py` metrics as everything else. The ensembles are fitted on the training split only (`fit_ensemble(model, Xtr, ytr)`); the test split is used once, afterwards, for evaluation. The "best individual model" is chosen from **train-only CV PR-AUC** (MLP 0.959 vs SVM 0.952 vs LR 0.807), not from test results. Checks in `results/ensemble_sanity_checks.json`: reconstructed components reproduce `tuned_results.csv` (max diff 1e-16); voting equals the mean of component probabilities; the stacking meta-model equals a manual out-of-fold refit on training data (diff 0.0).

**Run:** `python -m src.experiments --ensemble-only [--seeds N]` (needs the saved results; writes only new files) or the full `python -m src.experiments` (ensemble is the last stage).

**New files:** `results/ensemble_results.csv`, `ensemble_comparison.csv`, `ensemble_vs_best_individual.csv`, `ensemble_components.csv`, `model_comparison_with_ensembles.csv` (all 17 original rows + the 2 ensemble rows, with an `Origin` column), `ensemble_seed_runs.csv` / `ensemble_seed_summary.csv`, `ensemble_sanity_checks.json`, `ensemble_predictions.pkl`; plots `08_ensemble_metrics.png`, `09_ensemble_roc_pr_curves.png`, `10_ensemble_confusion_matrices.png`.

**Result (test set, seed 42, 96 positives):**

| Model | F1 | Recall | Precision | PR-AUC | ROC-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| MLP tuned (best by CV) | 0.896 | 0.938 | 0.857 | 0.973 | 0.996 | 15 | 6 |
| SVM tuned | 0.925 | 0.896 | 0.956 | 0.978 | 0.996 | 4 | 10 |
| Logistic Regression tuned | 0.736 | 0.667 | 0.821 | 0.825 | 0.963 | 14 | 32 |
| Soft Voting Ensemble | 0.942 | 0.927 | 0.957 | 0.972 | 0.993 | 4 | 7 |
| Stacking Ensemble | 0.937 | 0.927 | 0.947 | 0.974 | 0.994 | 5 | 7 |

*Observed:* at the 0.5 threshold both ensembles have the highest F1 and far fewer false positives than the MLP (4-5 vs 15) at the cost of one extra false negative (7 vs 6); against the SVM they find 3 more positives. They do **not** improve the threshold-free ranking metrics: PR-AUC and ROC-AUC are equal to or slightly below the best individual models (ROC-AUC 0.993/0.994 vs 0.996). Over 5 re-seedings of the stochastic parts (same split; MLP init, Platt CV, stacking folds), F1 was 0.947 +/- 0.007 (voting), 0.941 +/- 0.006 (stacking), 0.906 +/- 0.010 (MLP), SVM 0.925 (deterministic). *Interpretation / caution:* the F1 gain mostly reflects fewer MLP false positives and amounts to a few test cases (96 positives); it is one split, and the re-seeding does not capture test-sampling uncertainty, so we do not claim a statistically established improvement. *Limitations:* single split; equal voting weights and an untuned meta-model; Platt-scaled SVM probabilities; component hyperparameters were chosen on the same training data that stacking's out-of-fold predictions come from (the test set is unaffected).

## Balanced Synthetic-Data Experiment (separate extension)
Mentor-requested comparison of the six models on a generated, exactly balanced dataset (prediction performance and computational cost, 5 seeds, mean +/- sd). Not part of the paper replication. Run `python -m src.synthetic_experiment`; method, validation, results and limitations are in `SYNTHETIC_EXPERIMENT.md`; outputs in `results/synthetic/`.


## Synthetic optimizer and ensemble extensions

The optional extension evaluates optimizer/solver configurations and ensemble methods on the balanced synthetic dataset (seed 42). Run it with:

```bash
python -m src.synthetic_extensions
```

It writes results to `results/synthetic/extensions/`: `synthetic_optimizer_comparisons.csv`, `synthetic_ensemble_comparison.csv`, and `README.md`. The extension uses a stratified 80/20 synthetic split and three-fold stratified CV on the training split only to keep the additional experiment practical. The held-out synthetic test split is used only for final reporting. It compares MLP optimizers (Adam, SGD at two learning rates, L-BFGS), Logistic Regression solvers, Perceptron regularisation/step-size settings, and Soft Voting/Stacking ensembles. Ensemble components reuse hyperparameters selected on the seed-42 synthetic training data. These results are a seed-42 extension and should not be presented as five-seed averages.

## Individual customer prediction (terminal)

Run from the project root:

```bash
python -m src.individual_prediction
```

The script loads the tuned hyperparameters in `results/best_hyperparameters.json`, makes the same stratified 80/20 real-data split with seed 42 used by the main workflow, fits the tuned Logistic Regression, SVM and MLP pipelines on the 4,000-record real training split, and evaluates them on the untouched 1,000-record real test split. It saves the fitted pipelines under `models/individual_prediction_*.joblib` and writes the held-out metrics to `results/individual_prediction_holdout_metrics.csv`. It then prompts for the 12 input features in the dataset's expected order and prints all three classifications.

The input units/codes follow `data/Personal_Loan.csv` (Income, CCAvg and Mortgage use the dataset's $000s units; Education is 1/2/3; account flags are 0/1). Use plausible values consistent with the dataset. The terminal tool reports class labels only; it deliberately does not show uncalibrated model outputs as real-world acceptance probabilities. These predictions estimate whether a customer resembles historical customers who accepted the offer, not loan eligibility or approval.

Note: this command fits the models from the project's saved tuned hyperparameters rather than loading a pre-existing trained pipeline. The held-out test data is used only for evaluation and is not used to fit the models.
