# Model Evaluation & Comparison Report
## Knee Osteoarthritis Progression Prediction

| Model Name | ROC-AUC | Sensitivity (Recall) | Specificity | Precision | F1-Score | Brier Score | FN Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Logistic Regression` | 0.7049 | 0.3733 | 0.8800 | 0.6512 | 0.4746 | 0.2075 | 47 |
| `Random Forest` | 0.6401 | 0.1867 | 0.8800 | 0.4828 | 0.2692 | 0.2213 | 61 |
| `XGBoost` | 0.6190 | 0.3733 | 0.8000 | 0.5283 | 0.4375 | 0.2313 | 47 |

### Medical Performance Interpretation
- **Sensitivity / Recall**: Crucial for screening; false negatives represent high-risk progression patients missed by the model.
- **Specificity**: Minimizes unnecessary follow-up burden by reducing false positives.
- **ROC-AUC**: Evaluates discrimination quality across all decision thresholds.

> **Disclaimer**: High predictive metrics do NOT validate clinical deployability. Model predictions represent probabilistic risk estimates for research decision-support only.