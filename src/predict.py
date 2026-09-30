"""
Inference & Prediction Module
=============================
Loads serialized model pipeline(s) and computes progression risk probabilities,
risk categories, and top feature contribution breakdowns for input patient baseline data.

Supports selecting which trained model to use at prediction time (e.g. Logistic
Regression vs Random Forest vs XGBoost), when the serialized artifact contains
more than one fitted model.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from src.config import config
from src.explainability import compute_instance_shap


def load_model_pipeline(model_path: Path = config.MODELS_DIR / "trained_pipeline.pkl") -> Dict[str, Any]:
    """Load serialized model pipeline artifact (may contain one or several fitted models)."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please train a model first (python -m src.train).")
    return joblib.load(model_path)


def _normalize_pipeline_dict(pipeline_dict: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Return (models_by_name, default_name) regardless of whether the artifact was
    saved in the new multi-model format ({"models": {...}, "best_model_name": ...})
    or the older single-model format ({"preprocessor":..., "model":..., "model_name":...}).
    """
    if "models" in pipeline_dict:
        return pipeline_dict["models"], pipeline_dict.get("best_model_name", next(iter(pipeline_dict["models"])))

    # Legacy single-model artifact
    name = pipeline_dict.get("model_name", "Model")
    return {name: pipeline_dict}, name


def list_available_models(pipeline_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return [{name, metrics, is_default}, ...] for every model stored in the artifact."""
    models_by_name, default_name = _normalize_pipeline_dict(pipeline_dict)
    return [
        {"name": name, "metrics": entry.get("metrics", {}), "is_default": name == default_name}
        for name, entry in models_by_name.items()
    ]


def predict_patient_risk(
    patient_data: Dict[str, Any],
    pipeline_dict: Dict[str, Any] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predict progression risk probability and risk tier for a single patient record,
    using the requested model (falls back to the trained default if not specified
    or not found).

    Returns:
      - prob_progression: Probability between 0.0 and 1.0
      - risk_category: 'Low Risk', 'Moderate Risk', or 'Higher Risk'
      - feature_contributions: Top features influencing the risk estimate
      - model_used: Name of the model that produced this prediction
    """
    if pipeline_dict is None:
        pipeline_dict = load_model_pipeline()

    models_by_name, default_name = _normalize_pipeline_dict(pipeline_dict)
    chosen_name = model_name if model_name in models_by_name else default_name
    entry = models_by_name[chosen_name]

    preprocessor = entry["preprocessor"]
    model = entry["model"]
    feature_names = entry["feature_names"]

    df_patient = pd.DataFrame([patient_data])
    X_proc = preprocessor.transform(df_patient)
    prob_progression = float(model.predict_proba(X_proc)[0, 1])

    if prob_progression < config.LOW_RISK_THRESHOLD:
        risk_category = "Low Risk"
    elif prob_progression < config.HIGH_RISK_THRESHOLD:
        risk_category = "Moderate Risk"
    else:
        risk_category = "Higher Risk"

    # Feature contribution breakdown: real per-patient SHAP values when this
    # model's entry has a saved background sample (models trained after this
    # patch). Older artifacts fall back to the previous global-importance
    # behaviour, which is the SAME value for every patient and is flagged as
    # such via "personalized" below.
    background = entry.get("shap_background")
    feature_contributions = {}
    personalized = False
    if background is not None:
        try:
            shap_vals = compute_instance_shap(model, X_proc, background)
            for fname, val in zip(feature_names, shap_vals):
                feature_contributions[fname] = round(float(val), 4)
            personalized = True
        except Exception as e:
            print(f"[!] Per-patient SHAP explanation failed, falling back to global importances: {e}")

    if not personalized:
        if hasattr(model, 'feature_importances_'):
            for fname, imp in zip(feature_names, model.feature_importances_):
                feature_contributions[fname] = round(float(imp), 4)
        elif hasattr(model, 'coef_'):
            for fname, coef in zip(feature_names, model.coef_[0]):
                feature_contributions[fname] = round(float(coef), 4)

    return {
        "predicted_probability": round(prob_progression, 4),
        "risk_category": risk_category,
        "feature_contributions": feature_contributions,
        "model_used": chosen_name,
        "personalized": personalized,
    }