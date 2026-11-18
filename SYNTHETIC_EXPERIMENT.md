# Balanced Synthetic-Data Experiment

## Goal

This is a separate extension to the original personal-loan prediction paper. It compares six models on a newly generated, balanced synthetic dataset that learns from the original customers who accepted or rejected a personal loan.

The generator learns class-specific feature distributions and relationships from the **real training split only**. It then generates new customer rows with exactly 50% class 0 and 50% class 1. The generated labels are assigned conditionally on the historical loan-decision class, so this experiment models patterns in the old decisions; it does not establish causal reasons for loan acceptance or guarantee real-world approval outcomes.

## Run

```bash
pip install -r requirements.txt
python -m src.synthetic_experiment              # full run, seeds 42–46
python -m src.synthetic_experiment --quick       # smoke test, one seed and smaller dataset
python -m src.synthetic_experiment --report-only # rebuild reports from saved run tables
```

The verified full run used five seeds and took about 560.1 seconds (9.3 minutes) in this environment. Runtime varies by machine.

## Experimental procedure

1. Load `data/Personal_Loan.csv` (5,000 customers: 4,520 rejected and 480 accepted). Drop the ID column; retain the 12 predictor features and the `Personal Loan` target.
2. For each seed 42–46, create a stratified 80/20 real train/test split. Fit the generator on the real training split only; do not use the real test split to generate data or tune models.
3. Fit a class-conditional Gaussian copula generator. For each class, it learns the empirical marginal distribution of each feature and a dependence matrix built from rank-based normal scores. It generates 2,500 class-0 rows and 2,500 class-1 rows.
4. Generate `Experience` as `Age - Gap` and reject records whose generated `Experience` or `Age - Experience` gap falls outside the ranges observed in the real training split. Generated features are also checked against training-set ranges.
5. Split the 5,000 synthetic rows into an 80% training set and a 20% synthetic test set, stratified by label.
6. Train and tune Perceptron, Logistic Regression, KNN, SVM and MLP using five-fold stratified cross-validation on synthetic training data only. K-Means uses two clusters and is not tuned with a supervised objective. Feature scaling is inside the model pipeline to avoid leakage.
7. Evaluate each fitted model on the synthetic test set and separately on the untouched real test split (train on synthetic, test on real). Report metrics and computational cost.

## Generated dataset and validation

Each `results/synthetic/data/synthetic_balanced_seed*.csv` file contains 5,000 rows and 13 columns (12 predictors plus `Personal Loan`). The convenient copy `data/synthetic_balanced_personal_loan.csv` is the seed-42 dataset.

| Validation check | Result across the five-seed run |
|---|---:|
| Rows per synthetic dataset | 5,000 |
| Class balance | 2,500 class 0 / 2,500 class 1 |
| Rows outside observed training feature ranges | 0% |
| Age − Experience gap outside observed range | 0% |
| Duplicate rows within each synthetic dataset | 0 |
| Exact matches to rows in the original dataset | 0 |
| Mean marginal KS/TVD difference (synthetic vs real train) | 0.0125 |
| Reference mean KS/TVD (real test vs real train) | 0.0430 |
| Mean absolute Spearman correlation difference, class 0 | 0.0230 |
| Mean absolute Spearman correlation difference, class 1 | 0.0289 |
| Cohen’s-d effect-size correlation (real vs synthetic) | 0.9990 |
| Real-vs-synthetic detectability AUC, class 0 (0.5 = indistinguishable) | 0.644 |
| Real-vs-synthetic detectability AUC, class 1 (0.5 = indistinguishable) | 0.507 |

The generator reproduces the marginal distributions and feature-class effect sizes reasonably well, and it creates no exact row copies. However, the class-0 synthetic records remain detectably different from real class-0 records (AUC above 0.5). The synthetic data is therefore a useful controlled dataset, not a perfect replacement for real customers.

## Results: synthetic test set

The performance score is the unweighted mean of F1, balanced accuracy, ROC-AUC and PR-AUC (average precision). Values below are means over five seeds. Tuning time includes cross-validation and search refitting; final-fit time is measured separately.

| Model | Score | Score SD | F1 | Balanced acc. | ROC-AUC | PR-AUC | Tuning (s) | Final fit (s) | Inference (µs/record) | Size (KB) | Real-test ROC-AUC (TSTR) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Perceptron | 0.892 | 0.022 | 0.855 | 0.854 | 0.933 | 0.926 | 1.50 | 0.004 | 0.153 | 1.8 | 0.940 |
| Logistic Regression | 0.905 | 0.006 | 0.867 | 0.867 | 0.945 | 0.939 | 6.25 | 0.004 | 0.156 | 1.6 | 0.957 |
| K-Means | 0.511 | 0.010 | 0.532 | 0.491 | 0.511 | 0.512 | 0.00 | 0.014 | 0.301 | 17.5 | 0.538 |
| KNN | 0.907 | 0.004 | 0.867 | 0.870 | 0.946 | 0.945 | 4.07 | 0.004 | 58.018 | 495.2 | 0.969 |
| SVM | 0.923 | 0.002 | 0.893 | 0.892 | 0.957 | 0.952 | 17.14 | 0.157 | 22.032 | 120.4 | 0.981 |
| MLP | 0.922 | 0.002 | 0.890 | 0.888 | 0.956 | 0.952 | 73.74 | 0.982 | 0.402 | 42.7 | 0.981 |

### Interpretation

- **SVM has the highest average composite score** (0.923), followed closely by MLP (0.922). The difference is only 0.0018 and is smaller than the observed seed-to-seed variation, so this should be treated as a close result rather than proof that SVM is universally better.
- Under the pre-declared selection rule (within 0.01 of the best composite score, then lowest tuning + final-fit time), **SVM is the best balance**: it is close to MLP in predictive score and requires substantially less tuning time.
- **Logistic Regression** has the smallest serialized model and very low average inference time. **KNN** has a very short final fit but a larger model and slower inference because prediction depends on stored training examples.
- **K-Means is near chance level** on this classification task. It is unsupervised, and its two clusters do not align well with the loan-decision classes.
- On the untouched real test split, synthetic-trained SVM and MLP achieve mean ROC-AUC values of 0.981 and 0.981 respectively. These results indicate useful transfer of ranking patterns, but threshold-based precision is lower than on the balanced synthetic test set because real data has far fewer accepted-loan cases.

## Limitations and interpretation notes

- Only the original training split is used to fit the generator. However, the positive class has relatively few original examples, so the estimated class-1 dependence structure is based on limited data.
- The generator is class-conditional: it first learns what each historical class looks like and then generates rows for each class. This makes the balanced classification comparison possible, but it does not independently discover loan-acceptance labels from scratch.
- Balancing changes the class prior from about 9.6% positives in the original data to 50% in synthetic data. Accuracy and threshold-based precision/false-positive rates therefore should not be compared directly between the two test sets.
- MLP search is deliberately reduced to eight random configurations from a larger space to keep the five-seed run practical. Some MLP candidates reached the configured 200-iteration limit and emitted convergence warnings; the warning is retained in the run log and should be considered when interpreting its results.
- Efficiency measurements are wall-clock timings and can vary by hardware and system load. Small differences in microsecond-level inference timing are not meaningful without repeated controlled benchmarking.
- This project reproduces patterns in historical data, not causal explanations or a real bank’s current underwriting policy. It should not be used to make actual lending decisions.

## Output files

- `results/synthetic/data/`: five balanced generated CSV files.
- `results/synthetic/tables/final_comparison.csv`: main model comparison.
- `results/synthetic/tables/per_seed_performance.csv`: all per-seed synthetic and real test metrics.
- `results/synthetic/tables/per_seed_efficiency.csv`: per-seed timing and model-size measurements.
- `results/synthetic/tables/quality_summary.csv`: averaged synthetic-data quality checks.
- `results/synthetic/plots/`: quality, performance, confusion-matrix, ROC/PR and efficiency plots.
- `results/synthetic/run_log.txt`: full five-seed run log.
