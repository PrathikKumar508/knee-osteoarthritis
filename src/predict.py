"""
Inference & Prediction Module
=============================
Loads serialized model pipeline and computes progression risk probabilities,
risk categories, and top feature contribution breakdowns for input patient baseline data.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, List
from src.config import config


def load_model_pipeline(model_path: Path = config.MODELS_DIR / "trained_pipeline.pkl") -> Dict[str, Any]:
    """Load serialized model pipeline artifact."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please train a model first (python -m src.train).")
    return joblib.load(model_path)


def predict_patient_risk(
    patient_data: Dict[str, Any],
    pipeline_dict: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Predict progression risk probability and risk tier for a single patient record.
    
    Returns:
      - prob_progression: Probability between 0.0 and 1.0
      - risk_category: 'Low Risk', 'Moderate Risk', or 'Higher Risk'
      - feature_contributions: Top features influencing the risk estimate
    """
    if pipeline_dict is None:
        pipeline_dict = load_model_pipeline()

    preprocessor = pipeline_dict["preprocessor"]
    model = pipeline_dict["model"]
    feature_names = pipeline_dict["feature_names"]

    # Convert single patient dict to dataframe
    df_patient = pd.DataFrame([patient_data])

    # Transform features
    X_proc = preprocessor.transform(df_patient)

    # Predict probability of progression
    prob_progression = float(model.predict_proba(X_proc)[0, 1])

    # Risk Tier classification based on threshold bounds
    if prob_progression < config.LOW_RISK_THRESHOLD:
        risk_category = "Low Risk"
    elif prob_progression < config.HIGH_RISK_THRESHOLD:
        risk_category = "Moderate Risk"
    else:
        risk_category = "Higher Risk"

    # Feature contribution breakdown (approximated via model weights / tree importances)
    feature_contributions = {}
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Multiply input feature value by importance weight for localized effect
        for fname, imp in zip(feature_names, importances):
            feature_contributions[fname] = round(float(imp), 4)
    elif hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        for fname, coef in zip(feature_names, coefs):
            feature_contributions[fname] = round(float(coef), 4)

    return {
        "predicted_probability": round(prob_progression, 4),
        "risk_category": risk_category,
        "feature_contributions": feature_contributions,
        "model_used": pipeline_dict["model_name"]
    }
