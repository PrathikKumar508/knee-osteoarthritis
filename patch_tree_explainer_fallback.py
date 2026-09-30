from pathlib import Path

path = "src/explainability.py"
text = Path(path).read_text()

old = '''def compute_instance_shap(model, x_processed, background):
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
    return values.reshape(-1)'''

new = '''def compute_instance_shap(model, x_processed, background):
    """
    Per-instance SHAP contributions for the positive (progression) class,
    for ONE patient's already-preprocessed input row.

    Tree models are tried on the probability scale first. Some tree ensembles
    trained on one-hot encoded binary columns trip a SHAP limitation
    ("Categorical split is not yet supported") in that mode; when that
    happens we fall back to tree_path_dependent, which needs no background
    sample but returns values on the model's raw margin (log-odds-like)
    scale rather than probability. Logistic Regression is always explained
    on its linear (log-odds) scale. Magnitudes are therefore not directly
    comparable across model types, but the ranking and sign within one
    patient's explanation are valid regardless of which mode was used.
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
    return values.reshape(-1)'''

n = text.count(old)
if n != 1:
    raise SystemExit(f"Expected 1 match, found {n}. Paste 'sed -n \"1,60p\" src/explainability.py' so I can see the whole file.")
Path(path).write_text(text.replace(old, new))
print("patched src/explainability.py")
