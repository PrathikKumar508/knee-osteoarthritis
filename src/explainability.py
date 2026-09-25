"""
Model Explainability & SHAP Feature Analysis Module
===================================================
Provides SHAP (SHapley Additive exPlanations) analysis and tree feature importance plots
to explain model predictions. Provides associative, non-causal patient feature risk summaries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from typing import Dict, List, Any, Tuple
from src.config import config


def compute_shap_explanations(
    model: Any,
    X_processed: np.ndarray,
    feature_names: List[str],
    output_dir: Path = config.FIGURES_DIR
) -> Tuple[Any, np.ndarray]:
    """
    Compute SHAP values for tree-based models (Random Forest, XGBoost) or linear models.
    Saves SHAP summary beeswarm plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
        
        # Handle multi-class / binary array shapes
        if isinstance(shap_values, list):
            # Class 1 (progression) SHAP values
            vals = shap_values[1]
        elif len(shap_values.shape) == 3:
            vals = shap_values[:, :, 1]
        else:
            vals = shap_values

    except Exception as e:
        print(f"TreeExplainer failed, falling back to KernelExplainer / LinearExplainer: {e}")
        explainer = shap.Explainer(model, X_processed[:100])
        shap_values_obj = explainer(X_processed)
        vals = shap_values_obj.values
        if len(vals.shape) == 3:
            vals = vals[:, :, 1]

    # Plot SHAP summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals, X_processed, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance & Impact on OA Progression Risk", fontsize=14, pad=15)
    plt.tight_layout()
    plot_path = output_dir / "shap_summary_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return explainer, vals


def generate_patient_risk_explanation(
    patient_features: Dict[str, Any],
    feature_importances: Dict[str, float]
) -> List[str]:
    """
    Generate associative, non-causal human-readable feature explanation statements.
    
    IMPORTANT: Carefully worded to state association, NOT medical causality.
    """
    explanations = []

    sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)

    for feat_name, imp in sorted_features[:5]:
        val = patient_features.get(feat_name, 'N/A')
        direction = "increased" if imp > 0 else "decreased"
        
        statement = (
            f"**{feat_name}** (value: {val}): Associated with **{direction}** estimated probability of progression "
            f"(relative model importance weight: {abs(imp):.3f})."
        )
        explanations.append(statement)

    return explanations
