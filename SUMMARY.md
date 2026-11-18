# Experiment summary

Labels: **[Paper]** what the paper reports · **[Observed]** what our runs produced · **[Interpretation]** our reading.

## 1. Paper methodology
Kaggle Personal Loan data (5,000 customers, ~10% positive), 80/20 split, ID column removed, MinMaxScaler for the MLP, ROC/AUC as the main metric after noticing Perceptron and LogReg predict all zeros. Models: Perceptron, LogReg, K-Means (text) / KNeighborsClassifier (figure legend), linear SVM, MLP (relu, adam, constant LR, L2 1e-4; "100 hidden layers" - our interpretation: hidden_layer_sizes=(100,), see addendum). Reported AUCs: .61, .81, .75, .94, .99; MLP 98% accuracy, 9 FP / 11 FN. No seed, stratification or hyperparameter search is described.

## 2. Replication setup
Seed 42, stratified 80/20 (train 4000 with 384 positives; test 1000 with 96). Baselines use sklearn defaults except where the paper states otherwise. Two variants: paper-faithful and all-scaled.

## 3. Tuning methodology (extension)
GridSearchCV (Perceptron, LogReg, KNN, SVM) and RandomizedSearchCV (MLP: a LIMITED search of 30 random candidates from a space of 2,880 combinations, ~1%; not an exhaustive optimisation. A first attempt with a larger space (40 candidates incl. batch size 32 and max_iter 600) was abandoned for runtime before any MLP result was seen, and replaced by this one), 5-fold stratified CV on training data, refit on PR-AUC; all CV metrics recorded. Test set used once afterwards.

## 4. Metrics
Accuracy, precision, recall, F1, specificity, balanced accuracy, ROC-AUC, PR-AUC, FP, FN (positive class = loan accepted). Accuracy is misleading here because predicting "no one accepts" scores 90.4% on the test set (Perceptron baseline: accuracy 0.904, recall 0).

## 5. Results (test set, seed 42)
| Model | Version | Acc | Prec | Recall | F1 | ROC-AUC | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Perceptron | Baseline (paper) | .904 | 0 | 0 | 0 | .589 | .149 | 0 | 96 |
| Perceptron | Baseline (scaled) | .944 | .733 | .656 | .692 | .956 | .760 | 23 | 33 |
| Perceptron | Tuned | .924 | .576 | .792 | .667 | .952 | .763 | 56 | 20 |
| LogReg | Baseline (paper) | .932 | .694 | .521 | .595 | .959 | .722 | 22 | 46 |
| LogReg | Baseline (scaled) | .953 | .855 | .615 | .715 | .963 | .821 | 10 | 37 |
| LogReg | Tuned | .954 | .821 | .667 | .736 | .963 | .825 | 14 | 32 |
| K-Means | Baseline (paper) | .519 | .081 | .385 | .133 | .468 | .087 | 422 | 59 |
| KNeighbors | Baseline (paper) | .905 | .511 | .240 | .326 | .811 | .325 | 22 | 73 |
| KNeighbors | Baseline (scaled) | .952 | .914 | .552 | .688 | .941 | .813 | 5 | 43 |
| KNeighbors | Tuned | .931 | 1.0 | .281 | .439 | .978 | .891 | 0 | 69 |
| SVM | Baseline (paper, linear) | .916 | .620 | .323 | .425 | .939 | .529 | 19 | 65 |
| SVM | Baseline (scaled) | .954 | .891 | .594 | .712 | .958 | .825 | 7 | 39 |
| SVM | Tuned (RBF, C=10) | .986 | .956 | .896 | .925 | .996 | .978 | 4 | 10 |
| MLP | Baseline (paper = scaled) | .986 | .936 | .917 | .926 | .996 | .977 | 6 | 8 |
| MLP | Tuned | .979 | .857 | .938 | .896 | .996 | .973 | 15 | 6 |
Full table: `results/model_comparison.csv`; exact hyperparameters: `results/final_model_hyperparameters_all.json`.

## 6. Answers to the guiding questions
1. **Reproducible?** Partly. Perceptron (all zeros, AUC .59 vs .61), linear SVM (AUC .939 vs .94) and MLP (AUC .996, acc .986 vs .99/.98) are close. Logistic Regression is not (we get 72 positive predictions and AUC .959 vs .81; the cause is not established). K-Means is not (AUC .468 vs .75, accuracy .52 vs ~.90). KNeighbors is in the same range (.811 vs .75), not exact.
2. **Good baselines:** the scaled MLP stands out (F1 .926). Unscaled linear models and SVM have low recall (.32–.52).
3./4./5. **Tuning:** clear gain for SVM (F1 .712 -> .925, recall .594 -> .896, PR-AUC .825 -> .978). Small gain for LogReg. No gain for the MLP (F1 .926 -> .896; over 10 init seeds F1 .931 -> .909, FP 6.8 -> 11.6, FN unchanged). KNN improved ranking (PR-AUC .813 -> .891) but its default-threshold recall fell to .281. Perceptron traded precision for recall without improving PR-AUC. **Most baseline gaps are explained by scaling, not tuning.**
6. **Optimizers:** MLP CV PR-AUC: adam .957, SGD(lr 1e-2) .952, L-BFGS .901, SGD(lr 1e-3) .793. [Interpretation] SGD depends mostly on learning rate; Adam and tuned SGD are not distinguishable. LogReg solvers: .8030–.8033 except liblinear .798, i.e. essentially no solver effect on a convex objective. Perceptron configurations: CV PR-AUC .746–.777, CV recall .42–.76.
7. **Metrics vs accuracy:** yes. Accuracy ranks Perceptron (.904) and KNN (.905) close to LogReg (.932), hiding recall of 0, .24 and .52. KNN-tuned has accuracy .931 with recall .28.
8. **Sensitivity:** LogReg to C (very low C collapses PR-AUC), SVM to kernel/gamma, KNN to k (k=1 worst), Perceptron to alpha. MLP search is small and noisy; see `hyperparameter_sensitivity.csv`.
9. **Paper claims:** "MLP is the clear winner" holds for the paper-faithful baselines but not after SVM tuning (tuned SVM F1 .925 vs MLP .926; ROC-AUC .996 both). The paper's explanation (probabilistic models fail because they are probabilistic) is not supported: scaling alone brings Perceptron and LogReg to F1 ~.7. The claim that scaling is "the only change" needed for the MLP is supported in direction: over 10 seeds unscaled MLP gets recall .185 ± .32 and accuracy .825 ± .26 vs scaled recall .932, accuracy .987, though the paper's "~93% acc / .93 AUC" unscaled is not matched (we see AUC .84 on average and very unstable runs).
10. **Limitations:** one dataset copy (GitHub mirror) and one headline split; recall SE ~.03; baseline MLP and LogReg hit max_iter; reduced MLP search; fixed 0.5 threshold; no resampling (by design); K-Means label mapping is our choice; tuned-vs-baseline differences below ~0.02 F1 should not be read as real.

## 7. Differences vs paper: possible reasons (not tested individually)
Unknown seed/stratification; sklearn version (1.8 here, much newer than the paper's); K-Means label mapping; LogReg convergence; unreported preprocessing.

## 8. Verification addendum (see `results/verification/VERIFICATION_REPORT.md`)
* **"100 hidden layers" is an interpretation.** All other settings the paper lists equal sklearn's MLPClassifier defaults, whose hidden_layer_sizes is (100,), which is consistent with but does not prove our reading. Sensitivity (5 init seeds): (100,) F1 .928 / PR-AUC .978; (100,100) .914 / .973; (100,100,100) .906 / .966.
* **Logistic Regression discrepancy (paper AUC .81 / all zeros vs ours .959 / 72 positives) is unresolved.** Over 400 fits (10 variants x 20 splits x stratified/unstratified) no variant gave AUC near .81 (range .901-.979) and none predicted all zeros. Variants: max_iter 100/1000/10000, drop ZIP, include ID, Education one-hot, MinMax/Standard scaling, liblinear, C=1e6. Non-convergence at max_iter=100 lowers AUC by about .02 only. Our seed-42 value (.959) is slightly above the range seen over seeds 0-19 (.919-.956, stratified).
* **K-Means discrepancy (paper .75 vs ours .47) is unresolved.** Unscaled K-Means splits on ZIP Code (std 2122 vs <=102 for other features): the two clusters have positive rates .093 / .100, so AUC ~ .49. Literal random-data-point init (n_init=1) gives .487; arbitrary label mapping .500; MinMax scaling .467-.492. Dropping ZIP gives AUC .716 (.686-.752 over 20 splits) but FP/FN of ~169/66, not the paper's 7/72; the paper lists ZIP as an input, so this does not show what the authors did.
* **The paper's K-Means/KNN numbers are not mutually consistent.** 7 FP / 72 FN on a 1000-row test set with 85-110 positives implies AUC ~ .57-.67 for a single operating point, not .75. The ROC curve labelled KNeighborsClassifier has one interior vertex (~FPR .10, TPR .59 read by eye; implied AUC ~ .745), i.e. binary scores, which a default k=5 predict_proba curve would not produce. No KNN setting we tried (k = 1,3,5,15, scaled/unscaled, hard labels) lands near that vertex (closest: scaled k=1, TPR .70 / FPR .012).
* **Optimizer comparison:** the MLP runs use max_iter=500 (baseline MLP uses the default 200 and does not converge). SGD at lr 1e-3 vs 1e-2 differ in learning rate, so "SGD vs Adam" is confounded with learning rate. liblinear's small gap vs other LogReg solvers is only partly explained by intercept regularisation (intercept_scaling=1000 shrinks the max decision-function gap from .55 to .09; 1e6 is numerically unreliable).

## 9. Ensemble extension (NEW, not in the paper) - see README "Ensemble Learning Extension"
Soft voting and stacking of the tuned LR + SVM + MLP (no new search; same split and 0.5 threshold). **Observed (test, seed 42):** F1 .942 (voting) / .937 (stacking) vs .896 (MLP, best by train-only CV PR-AUC) and .925 (SVM); FP 4 / 5 vs 15 (MLP) and 4 (SVM); FN 7 / 7 vs 6 and 10; PR-AUC .972 / .974 vs .973 and .978; ROC-AUC .993 / .994 vs .996 for both. **Interpretation:** the ensembles improved threshold-dependent metrics (mainly by removing MLP false positives) but not the ranking metrics (PR-AUC, ROC-AUC); the differences are a few test cases on one split and are not shown to be statistically significant. Files: `results/ensemble_*.csv`, `model_comparison_with_ensembles.csv`, plots 08-10.
