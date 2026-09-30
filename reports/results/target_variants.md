# Sensitivity analysis: target definition and knee subgroup

Logistic Regression, 5 x 5-fold patient-grouped cross-validation. AUC values from different subgroups are not directly comparable because the populations differ. Look at the KL + clinical minus KL-only difference within each row.

| Variant | Knees | Patients | Progression rate | ROC-AUC KL only | ROC-AUC KL + clinical | Difference | PR-AUC KL only | PR-AUC KL + clinical | Folds better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Primary: baseline KL 0-3, KL rise >= 1 | 6,986 | 3,659 | 0.139 | 0.618 | 0.649 | +0.031 | 0.184 | 0.224 | 25/25 |
| Baseline KL 0-2 (early), KL rise >= 1 | 6,098 | 3,402 | 0.128 | 0.610 | 0.647 | +0.037 | 0.166 | 0.214 | 25/25 |
| Baseline KL 1-3, KL rise >= 1 | 4,144 | 2,582 | 0.179 | 0.548 | 0.601 | +0.052 | 0.197 | 0.244 | 24/25 |
| Baseline KL 2-3 (definite OA), KL rise >= 1 | 2,784 | 1,922 | 0.170 | 0.551 | 0.592 | +0.042 | 0.190 | 0.232 | 22/25 |
| Baseline KL 0-3, rise >= 1 and follow-up KL >= 2 | 6,986 | 3,659 | 0.125 | 0.671 | 0.705 | +0.034 | 0.187 | 0.229 | 25/25 |

Only the primary definition was chosen in advance. The other rows show how sensitive the conclusion is to that choice.