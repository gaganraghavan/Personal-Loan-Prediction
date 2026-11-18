# Synthetic balanced experiment: summary (mean over 5 seeds)

| Model | Performance score | F1 | Balanced Accuracy | ROC-AUC | PR-AUC | Performance score (std) | Tuning time (s) | Training time (s) | Tuning + training time (s) | Inference per record (us, batch avg) | Model size (KB) | Real-test ROC-AUC (mean, train-on-synthetic) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Perceptron | 0.892 | 0.8546 | 0.8544 | 0.9332 | 0.9259 | 0.02234 | 1.501 | 0.004449 | 1.505 | 0.1529 | 1.841 | 0.9399 |
| Logistic Regression | 0.9046 | 0.8669 | 0.8674 | 0.9449 | 0.9394 | 0.00571 | 6.25 | 0.004359 | 6.254 | 0.156 | 1.614 | 0.9575 |
| K-Means | 0.5115 | 0.5318 | 0.4908 | 0.5111 | 0.5121 | 0.009934 | 0 | 0.01353 | 0.01353 | 0.3012 | 17.5 | 0.5384 |
| KNN | 0.9071 | 0.8667 | 0.8702 | 0.9465 | 0.9451 | 0.004231 | 4.071 | 0.003837 | 4.075 | 58.02 | 495.2 | 0.969 |
| SVM | 0.9233 | 0.8928 | 0.8918 | 0.9569 | 0.9518 | 0.002326 | 17.14 | 0.1575 | 17.3 | 22.03 | 120.4 | 0.9809 |
| MLP | 0.9216 | 0.8898 | 0.8884 | 0.9564 | 0.9517 | 0.001793 | 73.74 | 0.9819 | 74.73 | 0.4021 | 42.74 | 0.9812 |

```json
{
 "seeds": 5,
 "performance_score_definition": "mean of F1, Balanced Accuracy, ROC-AUC, PR-AUC(average precision) on the synthetic test set",
 "best_predictive_performance": "SVM",
 "gap_to_second": 0.0018,
 "second": "MLP",
 "gap_smaller_than_seed_std": true,
 "per_metric_winners": {
  "F1": "SVM",
  "Balanced Accuracy": "SVM",
  "ROC-AUC": "SVM",
  "PR-AUC": "SVM",
  "Recall": "SVM",
  "Precision": "KNN",
  "Accuracy": "SVM"
 },
 "fastest_to_train_final_fit": "KNN",
 "fastest_total_tuning_plus_training": "K-Means",
 "fastest_inference_per_record": "Perceptron",
 "smallest_model": "Logistic Regression",
 "best_balance_rule": "among models with performance score within 0.01 of the best, the lowest mean tuning+training time",
 "best_balance_candidates": [
  "SVM",
  "MLP"
 ],
 "best_balance": "SVM",
 "pareto_front(perf up, tune+train time down, inference down)": [
  "Perceptron",
  "Logistic Regression",
  "K-Means",
  "KNN",
  "SVM",
  "MLP"
 ]
}
```
