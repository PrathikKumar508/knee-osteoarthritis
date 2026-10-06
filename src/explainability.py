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


def compute_instance_shap(model, x_processed, background):
    """
    Per-instance SHAP contributions for the positive (progression) class,
    for ONE patient's already-preprocessed input row.

    Returns (values, scale):
      - values: 1D array of per-feature contributions.
      - scale: "probability" or "log_odds", determined EMPIRICALLY rather
        than assumed from model type. Some tree ensembles trained on
        one-hot encoded binary columns trip a SHAP limitation ("Categorical
        split is not yet supported") when asked for probability-scale
        output; the fallback (tree_path_dependent) happens to still be on
        the probability scale for scikit-learn's RandomForestClassifier
        (its trees store class probabilities directly) but is on the raw
        margin / log-odds scale for gradient-boosted models like XGBoost.
        Rather than hardcode that per model type, we check whether the
        SHAP base value plus the sum of this patient's contributions
        reconstructs the model's own predicted probability; if it does,
        the scale is "probability", otherwise "log_odds". This is robust
        to whichever code path SHAP actually took.
    """
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, background)
        raw = explainer.shap_values(x_processed)
    else:
        try:
            explainer = shap.TreeExplainer(model, background, model_output="probability")
            raw = explainer.shap_values(x_processed)
        except Exception:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            raw = explainer.shap_values(x_processed)

    values = np.asarray(raw)
    if values.ndim == 3:
        values = values[..., -1]
    elif isinstance(raw, list):
        values = np.asarray(raw[-1])
    values = values.reshape(-1)

    base = explainer.expected_value
    if hasattr(base, "__len__"):
        base = np.asarray(base).reshape(-1)[-1]
    reconstructed = float(base) + float(values.sum())
    actual_prob = float(model.predict_proba(x_processed)[0, 1])
    scale = "probability" if abs(reconstructed - actual_prob) < 0.02 else "log_odds"

    return values, scale


def sample_background(X_processed, n=100, random_state=42):
    """A small reference sample of processed rows, saved per model so
    predict.py can explain new patients without needing the full training
    set at inference time."""
    rng = np.random.default_rng(random_state)
    if X_processed.shape[0] > n:
        idx = rng.choice(X_processed.shape[0], size=n, replace=False)
        return X_processed[idx]
    return X_processed


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