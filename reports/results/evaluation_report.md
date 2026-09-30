# Model Evaluation & Comparison Report
## Knee Osteoarthritis Progression Prediction

| Model Name | ROC-AUC | Sensitivity (Recall) | Specificity | Precision | F1-Score | Brier Score | FN Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Logistic Regression` | 0.6306 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1168 | 196 |
| `Random Forest` | 0.6172 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1183 | 196 |
| `XGBoost` | 0.6191 | 0.0000 | 0.9992 | 0.0000 | 0.0000 | 0.1188 | 196 |

### Medical Performance Interpretation
- **Sensitivity / Recall**: Crucial for screening; false negatives represent high-risk progression patients missed by the model.
- **Specificity**: Minimizes unnecessary follow-up burden by reducing false positives.
- **ROC-AUC**: Evaluates discrimination quality across all decision thresholds.

> **Disclaimer**: High predictive metrics do NOT validate clinical deployability. Model predictions represent probabilistic risk estimates for research decision-support only.