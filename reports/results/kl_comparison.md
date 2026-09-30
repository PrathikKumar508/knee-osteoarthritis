# KL grade only vs. baseline clinical features

Repeated patient-grouped cross-validation (5 x 5-fold). Knees: 6,986, patients: 3,659, progression rate: 0.139.

Reference values: a useless model has ROC-AUC 0.5, PR-AUC equal to the progression rate (0.139), and Brier score 0.120.

| Model | Features | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- |
| Logistic Regression | Clinical only (no KL) | 0.618 +/- 0.021 | 0.202 +/- 0.019 | 0.118 +/- 0.001 |
| Logistic Regression | KL + clinical | 0.649 +/- 0.023 | 0.224 +/- 0.024 | 0.116 +/- 0.002 |
| Logistic Regression | KL grade only | 0.618 +/- 0.023 | 0.184 +/- 0.013 | 0.117 +/- 0.001 |
| Random Forest | Clinical only (no KL) | 0.608 +/- 0.020 | 0.197 +/- 0.019 | 0.118 +/- 0.001 |
| Random Forest | KL + clinical | 0.644 +/- 0.022 | 0.220 +/- 0.022 | 0.116 +/- 0.001 |
| Random Forest | KL grade only | 0.618 +/- 0.023 | 0.184 +/- 0.013 | 0.117 +/- 0.001 |
| XGBoost | Clinical only (no KL) | 0.589 +/- 0.023 | 0.187 +/- 0.017 | 0.120 +/- 0.002 |
| XGBoost | KL + clinical | 0.625 +/- 0.021 | 0.208 +/- 0.019 | 0.118 +/- 0.002 |
| XGBoost | KL grade only | 0.618 +/- 0.023 | 0.184 +/- 0.013 | 0.117 +/- 0.001 |

Paired difference in ROC-AUC (KL + clinical minus KL only), same folds:

- Logistic Regression: mean +0.031, clinical model better in 25 of 25 folds
- Random Forest: mean +0.026, clinical model better in 25 of 25 folds
- XGBoost: mean +0.007, clinical model better in 13 of 25 folds

The +/- values are the spread across folds, not a confidence interval.