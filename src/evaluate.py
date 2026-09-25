"""
Model Evaluation & Medical Performance Metrics Module
=====================================================
Calculates comprehensive medical prediction metrics including ROC-AUC, Sensitivity (Recall),
Specificity, Precision, F1-Score, Confusion Matrix, and Brier Calibration Score.
Saves visual plots and tabular evaluation reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss
)
from src.config import config


def compute_medical_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate comprehensive binary classification metrics.
    
    Metrics:
      - ROC-AUC: Ability to rank high-risk patients above low-risk ones
      - Sensitivity / Recall: True Positive Rate (proportion of progressors correctly identified)
      - Specificity: True Negative Rate (proportion of non-progressors correctly identified)
      - Precision: Positive Predictive Value
      - F1-Score: Harmonic mean of Precision and Sensitivity
      - Brier Score: Measure of probability calibration quality (lower is better)
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    brier = brier_score_loss(y_true, y_prob)

    metrics = {
        "ROC_AUC": round(float(auc), 4),
        "Sensitivity_Recall": round(float(sensitivity), 4),
        "Specificity": round(float(specificity), 4),
        "Precision": round(float(precision), 4),
        "F1_Score": round(float(f1), 4),
        "Brier_Score": round(float(brier), 4),
        "True_Positives": int(tp),
        "False_Positives": int(fp),
        "True_Negatives": int(tn),
        "False_Negatives": int(fn)
    }

    return metrics


def plot_evaluation_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: Path = config.FIGURES_DIR
) -> Dict[str, str]:
    """Generate and save ROC Curve, Precision-Recall Curve, and Confusion Matrix plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, color='#1f77b4', lw=2, label=f'{model_name} (AUC = {auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0].set_ylabel('True Positive Rate (Sensitivity)')
    axes[0].set_title(f'ROC Curve - {model_name}')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # 2. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(rec, prec, color='#2ca02c', lw=2, label=f'{model_name}')
    axes[1].set_xlabel('Recall (Sensitivity)')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall Curve - {model_name}')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)

    # 3. Confusion Matrix Heatmap
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=['No Progression', 'Progression'],
                yticklabels=['No Progression', 'Progression'])
    axes[2].set_xlabel('Predicted Risk Class')
    axes[2].set_ylabel('Actual Outcome')
    axes[2].set_title(f'Confusion Matrix - {model_name}')

    plt.tight_layout()
    plot_path = output_dir / f"evaluation_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    saved_files["eval_plot"] = str(plot_path)

    return saved_files


def generate_evaluation_report(
    results_dict: Dict[str, Dict[str, float]],
    output_path: Path = config.RESULTS_DIR / "evaluation_report.md"
) -> str:
    """Generate markdown table comparing performance of multiple models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# Model Evaluation & Comparison Report",
        "## Knee Osteoarthritis Progression Prediction",
        "",
        "| Model Name | ROC-AUC | Sensitivity (Recall) | Specificity | Precision | F1-Score | Brier Score | FN Count |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    ]

    for model_name, metrics in results_dict.items():
        line = (
            f"| `{model_name}` | {metrics['ROC_AUC']:.4f} | {metrics['Sensitivity_Recall']:.4f} | "
            f"{metrics['Specificity']:.4f} | {metrics['Precision']:.4f} | {metrics['F1_Score']:.4f} | "
            f"{metrics['Brier_Score']:.4f} | {metrics['False_Negatives']} |"
        )
        report_lines.append(line)

    report_lines.extend([
        "",
        "### Medical Performance Interpretation",
        "- **Sensitivity / Recall**: Crucial for screening; false negatives represent high-risk progression patients missed by the model.",
        "- **Specificity**: Minimizes unnecessary follow-up burden by reducing false positives.",
        "- **ROC-AUC**: Evaluates discrimination quality across all decision thresholds.",
        "",
        "> **Disclaimer**: High predictive metrics do NOT validate clinical deployability. Model predictions represent probabilistic risk estimates for research decision-support only."
    ])

    report_content = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return report_content
