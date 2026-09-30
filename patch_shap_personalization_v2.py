"""
Patch script v2: matches the current multi-model shape of train.py / predict.py
(the "models": {...}, "best_model_name": ... artifact, with per-model
preprocessor / model / feature_names entries).

Adds a per-model SHAP background sample at training time, and uses it at
prediction time to compute per-patient SHAP contributions for whichever
model was selected, instead of the old fixed global importances.

Run once from the project root (~/KOA):
    python3 patch_shap_personalization_v2.py
    python3 -m src.train
"""

from pathlib import Path


def rep(path, old, new, count=1):
    text = Path(path).read_text()
    n = text.count(old)
    if n != count:
        raise SystemExit(
            f"Expected {count} match(es) of a block in {path}, found {n}.\n"
            f"Nothing was modified in {path}. Paste me the relevant lines of "
            f"the current file and I'll adjust the patch."
        )
    Path(path).write_text(text.replace(old, new))
    print(f"  patched {path}")


# src/explainability.py already has compute_instance_shap / sample_background
# from the previous patch run. If this script is used on a fresh checkout that
# does NOT have them yet, add them first (skipped here if already present).
expl_text = Path("src/explainability.py").read_text()
if "def compute_instance_shap" not in expl_text:
    rep(
        "src/explainability.py",
        "def compute_shap_explanations(",
        '''def compute_instance_shap(model, x_processed, background):
    """Per-instance SHAP contributions for the positive (progression) class."""
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
    """A small reference sample of processed rows, saved per model."""
    rng = np.random.default_rng(random_state)
    if X_processed.shape[0] > n:
        idx = rng.choice(X_processed.shape[0], size=n, replace=False)
        return X_processed[idx]
    return X_processed


def compute_shap_explanations(''',
    )
else:
    print("  src/explainability.py already has compute_instance_shap/sample_background, skipping")

# src/train.py: import sample_background if not already imported
train_text = Path("src/train.py").read_text()
if "sample_background" not in train_text.split("\n")[1] and "import sample_background" not in train_text:
    rep(
        "src/train.py",
        "from src.explainability import compute_shap_explanations",
        "from src.explainability import compute_shap_explanations, sample_background",
    )
else:
    print("  src/train.py already imports sample_background, skipping import patch")

# src/train.py: add a background sample to EACH model's serialized entry
# (each model has its own preprocessor and X_train_proc in this codebase)
rep(
    "src/train.py",
    '''        serialized_models[name] = {
            "preprocessor": preprocessor,
            "model": model,
            "model_name": name,
            "feature_names": feature_names,
            "metrics": metrics
        }''',
    '''        serialized_models[name] = {
            "preprocessor": preprocessor,
            "model": model,
            "model_name": name,
            "feature_names": feature_names,
            "metrics": metrics,
            # Small reference sample used to explain individual patients later
            # (see src/explainability.py: compute_instance_shap).
            "shap_background": sample_background(X_train_proc),
        }''',
)

# src/predict.py: use per-patient SHAP for whichever model was selected
rep(
    "src/predict.py",
    "from src.config import config",
    "from src.config import config\nfrom src.explainability import compute_instance_shap",
)
rep(
    "src/predict.py",
    '''    feature_contributions = {}
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
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
        "model_used": chosen_name
    }''',
    '''    # Feature contribution breakdown: real per-patient SHAP values when this
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
    }''',
)

# app/app.py: a recommendation line built from this patient's real top SHAP
# drivers (skipped if already applied)
app_text = Path("app/app.py").read_text()
if "generate_personalized_driver_summary" not in app_text:
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
    """One sentence naming THIS patient's largest model drivers, from real
    per-patient SHAP values. Returns "" if no personalized explanation is
    available, rather than showing a generic ranking as if it were specific
    to this patient."""
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
else:
    print("  app/app.py already patched, skipping")

print("\nAll patches applied.")
