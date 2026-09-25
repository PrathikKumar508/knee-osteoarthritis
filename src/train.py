"""
Model Training & Patient-Grouped Cross-Validation Module
=========================================================
Trains baseline Logistic Regression, Random Forest, and XGBoost models.
Enforces patient-grouped split (GroupShuffleSplit) to prevent data leakage across subjects.
Fits preprocessor strictly on train set and serializes best model pipeline.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.config import config, OAIConfig
from src.data_preprocessing import build_preprocessing_pipeline, get_feature_names_out
from src.feature_engineering import create_progression_target, enforce_baseline_features_only
from src.evaluate import compute_medical_metrics, plot_evaluation_curves, generate_evaluation_report
from src.explainability import compute_shap_explanations


def create_demo_synthetic_dataset(num_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic schema-compliant OAI baseline dataset strictly for unit testing
    and demo execution prior to real OAI file placement in data/raw/.
    """
    np.random.seed(config.RANDOM_STATE)
    patient_ids = [f"P{i:05d}" for i in np.random.randint(10000, 99999, size=num_samples)]
    
    data = {
        config.PATIENT_ID_COL: patient_ids,
        "V00AGE": np.random.normal(61, 8, num_samples).clip(45, 80).round(1),
        "V00BMI": np.random.normal(28.5, 4.5, num_samples).clip(18, 45).round(1),
        "V00WOMKP": np.random.randint(0, 20, num_samples),
        "V00WOMAD": np.random.randint(0, 68, num_samples),
        "V00WOMST": np.random.randint(0, 8, num_samples),
        "V00PASE": np.random.normal(150, 45, num_samples).clip(30, 350).round(1),
        "V00SEX": np.random.choice([1, 2], num_samples, p=[0.42, 0.58]),
        "V00KL": np.random.choice([0, 1, 2, 3], num_samples, p=[0.25, 0.35, 0.30, 0.10]),
        "V00INJ": np.random.choice([0, 1], num_samples, p=[0.75, 0.25]),
        "V00SURG": np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
    }

    df = pd.DataFrame(data)

    # Progression probability correlated with risk factors (Age, BMI, WOMAC pain, Baseline KL)
    # Clinically calibrated for ~25-35% 48-month progression rate
    log_odds = -1.2 + 0.04 * (df["V00AGE"] - 60) + 0.08 * (df["V00BMI"] - 28) + 0.05 * (df["V00WOMKP"] - 6) + 0.6 * (df["V00KL"] - 1) + 0.4 * df["V00INJ"]
    prob_prog = 1 / (1 + np.exp(-log_odds))
    
    df[config.FOLLOWUP_KL_COL] = df["V00KL"] + np.random.binomial(1, prob_prog)
    df[config.FOLLOWUP_KL_COL] = df[config.FOLLOWUP_KL_COL].clip(0, 4)

    return df


def train_models(df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """
    Execute end-to-end model training, validation, and evaluation pipeline.
    """
    print("=" * 70)
    print("  STAGE 5 & 6: MODEL TRAINING & PATIENT-GROUPED SPLITTING")
    print("=" * 70)

    # Stage 2: Create Progression Target
    if config.TARGET_COL not in df.columns:
        df = create_progression_target(df)

    # Extract X, y, and Patient Group IDs
    X, y, groups = enforce_baseline_features_only(
        df, input_features=config.get_feature_list()
    )

    # Patient-grouped train/test split (avoids patient leakage across splits)
    gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    # Verify zero patient overlap
    train_patients = set(groups_train.unique())
    test_patients = set(groups_test.unique())
    overlap = train_patients.intersection(test_patients)
    print(f"\n[Leakage Check] Patient Overlap between Train and Test splits: {len(overlap)} (Expected: 0)")
    assert len(overlap) == 0, "CRITICAL ERROR: Patient leakage detected between train and test sets!"

    # Stage 4: Fit Preprocessing Pipeline strictly on X_train
    preprocessor = build_preprocessing_pipeline(
        num_features=[c for c in config.NUMERICAL_FEATURES if c in X.columns],
        cat_features=[c for c in config.CATEGORICAL_FEATURES if c in X.columns]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = get_feature_names_out(
        preprocessor,
        num_features=[c for c in config.NUMERICAL_FEATURES if c in X.columns],
        cat_features=[c for c in config.CATEGORICAL_FEATURES if c in X.columns]
    )

    # Stage 5 & 6: Train Candidate Models
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=config.RANDOM_STATE)
    }

    if XGB_AVAILABLE:
        candidate_models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=config.RANDOM_STATE
        )

    eval_results = {}
    fitted_models = {}

    for name, model in candidate_models.items():
        print(f"\nTraining '{name}'...")
        model.fit(X_train_proc, y_train)
        
        y_prob = model.predict_proba(X_test_proc)[:, 1]
        metrics = compute_medical_metrics(y_test.values, y_prob)
        eval_results[name] = metrics
        fitted_models[name] = model

        print(f"  ROC-AUC: {metrics['ROC_AUC']:.4f} | Recall: {metrics['Sensitivity_Recall']:.4f} | Spec: {metrics['Specificity']:.4f} | F1: {metrics['F1_Score']:.4f}")
        plot_evaluation_curves(y_test.values, y_prob, name)

    # Save Evaluation Comparison Report
    generate_evaluation_report(eval_results)

    # Select Best Model based on ROC-AUC
    best_model_name = max(eval_results, key=lambda k: eval_results[k]["ROC_AUC"])
    best_model = fitted_models[best_model_name]
    print(f"\n[Best Model Selected]: '{best_model_name}' (ROC-AUC: {eval_results[best_model_name]['ROC_AUC']:.4f})")

    # Stage 8: Compute SHAP Explanations for Best Model
    try:
        print("\nComputing SHAP explanations...")
        compute_shap_explanations(best_model, X_test_proc, feature_names)
    except Exception as e:
        print(f"SHAP explanation skipped: {e}")

    # Serialize Full Pipeline (Preprocessor + Best Model)
    full_pipeline = {
        "preprocessor": preprocessor,
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": feature_names,
        "config": config,
        "metrics": eval_results[best_model_name]
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = config.MODELS_DIR / "trained_pipeline.pkl"
    joblib.dump(full_pipeline, model_save_path)
    print(f"Trained model pipeline serialized to: {model_save_path}")

    return best_model, eval_results


if __name__ == "__main__":
    processed_path = config.PROCESSED_DATA_DIR / "oai_processed.csv"
    if processed_path.exists():
        print(f"Loading processed dataset from {processed_path}...")
        df = pd.read_csv(processed_path)
    else:
        print("No processed OAI file found. Generating synthetic demo dataset...")
        df = create_demo_synthetic_dataset(num_samples=1000)

    train_models(df)
