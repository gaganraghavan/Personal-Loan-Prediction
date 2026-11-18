# Final Project Verification

Verification was performed after correcting the synthetic generator so that generated `Experience` values remain within the range observed in the real training split.

## Checks completed

| Check | Result |
|---|---|
| Python syntax/import compilation (`python -m compileall -q src`) | PASS |
| Quick synthetic experiment (`python -m src.synthetic_experiment --quick`) | PASS; generated the report and completed all six models |
| Full synthetic experiment (`python -m src.synthetic_experiment`) | PASS; completed five seeds (42–46), exit code 0, runtime about 560 seconds |
| Balanced row count for each full synthetic dataset | PASS; 5,000 rows, 2,500 per class |
| Feature count and target column | PASS; 12 predictor features plus `Personal Loan` |
| Generated rows outside original training feature ranges | PASS; 0 rows for each seed |
| `Age - Experience` outside the observed training range | PASS; 0 rows for each seed |
| Duplicate rows within generated datasets | PASS; 0 for each seed |
| Exact full-row matches to original dataset records | PASS; 0 for each seed |
| Seed-42 reproducibility | PASS; regenerated output exactly equals the saved seed-42 CSV |
| Model performance results | PASS; 60 evaluation rows (six models × five seeds × two evaluation sets), no missing metric values |
| Efficiency results | PASS; 30 rows (six models × five seeds), no missing values |

## Important interpretation notes

- The generator is trained on the real training split only; the real test split is not used to generate or tune data. The real test split is used for the separately reported train-on-synthetic/test-on-real check.
- The synthetic labels are assigned conditionally on historical loan-decision classes. This is appropriate for studying whether models can learn the patterns found in the historical data, but it does not establish causal reasons for approval or rejection.
- The full run emitted scikit-learn MLP convergence warnings for some configurations that reached the configured 200-iteration limit. These warnings are retained in `results/synthetic/run_log.txt` and described in `SYNTHETIC_EXPERIMENT.md`.
- The original paper-replication and ensemble outputs were preserved; the new synthetic experiment outputs are kept separately under `results/synthetic/`.
