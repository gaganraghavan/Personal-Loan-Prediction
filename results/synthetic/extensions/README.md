# Synthetic optimizer and ensemble extension

Dataset: balanced synthetic seed 42; 5000 rows; class counts: {0: 2500, 1: 2500}. Split: stratified 80/20. Three-fold stratified CV is performed on the synthetic training split only; the synthetic test split is used for final metrics.

## Optimizer/solver comparison

The best configuration by training-only CV PR-AUC in this run was **MLP Adam (lr=0.001)** (CV average precision 0.9481). See `synthetic_optimizer_comparisons.csv` for all tested configurations, test metrics, timing, and convergence warnings.

## Ensemble comparison

The best ensemble by training-only CV PR-AUC was **Soft Voting Ensemble** (CV average precision 0.9545). Soft voting averages the predicted probabilities of Logistic Regression, SVM, and MLP. Stacking trains a logistic-regression meta-model on out-of-fold base-model probabilities. See `synthetic_ensemble_comparison.csv` for the ensemble and component results.

## Interpretation and limitations

These are extension experiments on the seed-42 synthetic dataset, not five-seed aggregate results. Optimizer/solver settings are compared using the same split and CV folds. Ensemble component hyperparameters come from the synthetic experiment's seed-42 training-only search. Results may vary with the seed and hardware. Labels in the synthetic data are assigned conditionally from the original historical classes; this is a model-comparison experiment, not an independent loan-decision simulator. MLP convergence warnings are recorded rather than silently discarded.
