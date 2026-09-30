"""
Patch script: makes feature contributions genuinely per-patient using SHAP,
instead of the fixed global importances every patient currently gets.

Run once from the project root (~/KOA):
    python3 patch_shap_personalization.py

Then retrain so the saved pipeline includes a SHAP background sample:
    python3 -m src.train

After that, app/app.py's existing feature-contribution chart and the new
personalized recommendation line will use real per-patient SHAP values.
"""

from pathlib import Path


def rep(path, old, new, count=1):
    text = Path(path).read_text()
    n = text.count(old)
    if n != count:
        raise SystemExit(
            f"Expected {count} match(es) of a block in {path}, found {n}.\n"
            f"The file may already be patched, or has changed since this "
            f"script was written. Nothing was modified in {path}."
        )
    Path(path).write_text(text.replace(old, new))
    print(f"  patched {path}")


# 1. src/explainability.py: add the per-instance SHAP helper functions
rep(
    "src/explainability.py",
    "def compute_shap_explanations(",
    '''def compute_instance_shap(model, x_processed, background):
    """
    Per-instance SHAP contributions for the positive (progression) class,
    for ONE patient's already-preprocessed input row.

    Tree models (Random Forest, XGBoost) are explained on the probability
    scale. Logistic Regression is explained on its linear (log-odds) scale,
    so magnitudes are not directly comparable across model types, but the
    ranking and sign within one patient's explanation are valid.
    """
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.TreeExplainer(model, background, model_output="probability")

    raw = explainer.shap_values(x_processed)
    values = np.asarray(raw)
    if values.ndim == 3:
        values = values[..., -1]
    elif isinstance(raw, list):
        values = np.asarray(raw[-1])
    return values.reshape(-1)


def sample_background(X_processed, n=100, random_state=42):
    """A small reference sample of processed rows, saved with the trained
    pipeline so predict.py can explain new patients without needing the
    full training set at inference time."""
    rng = np.random.default_rng(random_state)
    if X_processed.shape[0] > n:
        idx = rng.choice(X_processed.shape[0], size=n, replace=False)
        return X_processed[idx]
    return X_processed


def compute_shap_explanations(''',
)

# 2. src/train.py: save a background sample with the pipeline
rep(
    "src/train.py",
    "from src.explainability import compute_shap_explanations",
    "from src.explainability import compute_shap_explanations, sample_background",
)
rep(
    "src/train.py",
    '''    full_pipeline = {
        "preprocessor": preprocessor,
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": feature_names,
        "config": config,
        "metrics": eval_results[best_model_name]
    }''',
    '''    full_pipeline = {
        "preprocessor": preprocessor,
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": feature_names,
        "config": config,
        "metrics": eval_results[best_model_name],
        # Small reference sample used to explain individual patients later
        # (see src/explainability.py: compute_instance_shap).
        "shap_background": sample_background(X_train_proc),
    }''',
)

# 3. src/predict.py: use per-patient SHAP instead of the global importances
rep(
    "src/predict.py",
    "from src.config import config",
    "from src.config import config\nfrom src.explainability import compute_instance_shap",
)
rep(
    "src/predict.py",
    '''    # Feature contribution breakdown (approximated via model weights / tree importances)
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
    }''',
    '''    # Feature contribution breakdown: real per-patient SHAP values when the
    # pipeline has a saved background sample (pipelines trained after this
    # patch). Older pipelines fall back to the previous behaviour, which is
    # the SAME value for every patient and is flagged as such below.
    background = pipeline_dict.get("shap_background")
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
        "model_used": pipeline_dict["model_name"],
        "personalized": personalized,
    }''',
)

# 4. app/app.py: a recommendation line built from this patient's real top
#    SHAP drivers, shown alongside the existing rule-based cards.
rep(
    "app/app.py",
    '''def generate_clinical_recommendations(prob: float, patient_data: dict) -> list:
    """Generate evidence-based monitoring and lifestyle considerations based on predicted risk."""
    recs = []
    ''',
    '''FRIENDLY_FEATURE_NAMES = {
    "V00AGE": "age", "V00BMI": "BMI", "V00WOMKP": "WOMAC pain score",
    "V00WOMAD": "WOMAC disability score", "V00WOMST": "WOMAC stiffness score",
    "V00PASE": "physical activity level", "V00SEX_1": "sex", "V00SEX_2": "sex",
    "V00KL_0": "baseline KL grade", "V00KL_1": "baseline KL grade",
    "V00KL_2": "baseline KL grade", "V00KL_3": "baseline KL grade",
    "V00INJ_0": "injury history", "V00INJ_1": "injury history",
    "V00SURG_0": "surgery history", "V00SURG_1": "surgery history",
}


def generate_personalized_driver_summary(feature_contributions: dict, personalized: bool, top_n: int = 3) -> str:
    """
    One sentence naming THIS patient's largest model drivers, built from
    real per-patient SHAP values. Returns "" if no personalized explanation
    is available (older pipeline), rather than showing a misleading generic
    ranking as if it were specific to this patient.
    """
    if not personalized or not feature_contributions:
        return ""

    ranked = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    parts = []
    for fname, val in ranked:
        label = FRIENDLY_FEATURE_NAMES.get(fname, fname)
        direction = "raised" if val > 0 else "lowered"
        parts.append(f"{label} ({direction} the estimate)")

    return (
        "For this patient specifically, the largest contributors to the model's estimate were: "
        + ", ".join(parts) + ". This ranking is computed per patient and will differ for other patients "
        "with the same risk tier."
    )


def generate_clinical_recommendations(prob: float, patient_data: dict) -> list:
    """Generate evidence-based monitoring and lifestyle considerations based on predicted risk."""
    recs = []
    ''',
)

print("\nAll patches applied.")
