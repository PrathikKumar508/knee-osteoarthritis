"""
Does baseline clinical data add predictive value beyond baseline KL grade?

Repeated patient-grouped cross-validation on data/processed/oai_processed.csv.
Compares feature sets (KL only, clinical only without KL, KL + clinical) with
Logistic Regression and a small Random Forest. Both knees of a patient always
stay in the same fold.

Run from the project root:
    python3 -m src.compare_models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import config
from src.data_preprocessing import build_preprocessing_pipeline
from src.feature_engineering import create_progression_target

N_SPLITS = 5
N_REPEATS = 5

NUM = list(config.NUMERICAL_FEATURES)
CAT = list(config.CATEGORICAL_FEATURES)
CAT_NO_KL = [c for c in CAT if c != "V00KL"]

FEATURE_SETS = {
    "KL grade only": ([], ["V00KL"]),
    "Clinical only (no KL)": (NUM, CAT_NO_KL),
    "KL + clinical": (NUM, CAT),
}

MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=20,
        random_state=config.RANDOM_STATE, n_jobs=-1),
    "XGBoost": lambda: XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        eval_metric="logloss", random_state=config.RANDOM_STATE, n_jobs=-1),
}


def make_pipeline(num, cat, model_name):
    return Pipeline([
        ("prep", build_preprocessing_pipeline(num, cat)),
        ("clf", MODELS[model_name]()),
    ])


def main():
    path = config.PROCESSED_DATA_DIR / "oai_processed.csv"
    df = create_progression_target(pd.read_csv(path))
    y = df[config.TARGET_COL].values
    groups = df[config.PATIENT_ID_COL].values
    prevalence = y.mean()
    # Brier score of a model that always predicts the overall progression rate
    brier_ref = brier_score_loss(y, np.full(len(y), prevalence))
    print(f"\nKnees: {len(df):,} | Patients: {len(np.unique(groups)):,} | "
          f"Progression rate: {prevalence:.3f}")

    records = []
    for repeat in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=config.RANDOM_STATE + repeat)
        for fold, (tr, te) in enumerate(cv.split(df, y, groups)):
            for fs_name, (num, cat) in FEATURE_SETS.items():
                cols = num + cat
                for model_name in MODELS:
                    pipe = make_pipeline(num, cat, model_name)
                    pipe.fit(df.iloc[tr][cols], y[tr])
                    p = pipe.predict_proba(df.iloc[te][cols])[:, 1]
                    records.append({
                        "repeat": repeat, "fold": fold,
                        "features": fs_name, "model": model_name,
                        "ROC_AUC": roc_auc_score(y[te], p),
                        "PR_AUC": average_precision_score(y[te], p),
                        "Brier": brier_score_loss(y[te], p),
                    })
        print(f"  finished repeat {repeat + 1}/{N_REPEATS}")

    res = pd.DataFrame(records)
    summary = (res.groupby(["model", "features"])[["ROC_AUC", "PR_AUC", "Brier"]]
                  .agg(["mean", "std"]).round(3))

    lines = [
        "# KL grade only vs. baseline clinical features",
        "",
        f"Repeated patient-grouped cross-validation ({N_REPEATS} x {N_SPLITS}-fold). "
        f"Knees: {len(df):,}, patients: {len(np.unique(groups)):,}, "
        f"progression rate: {prevalence:.3f}.",
        "",
        f"Reference values: a useless model has ROC-AUC 0.5, PR-AUC equal to the "
        f"progression rate ({prevalence:.3f}), and Brier score {brier_ref:.3f}.",
        "",
        "| Model | Features | ROC-AUC | PR-AUC | Brier |",
        "| --- | --- | --- | --- | --- |",
    ]
    for (model_name, fs_name), row in summary.iterrows():
        lines.append(
            f"| {model_name} | {fs_name} | "
            f"{row[('ROC_AUC', 'mean')]:.3f} +/- {row[('ROC_AUC', 'std')]:.3f} | "
            f"{row[('PR_AUC', 'mean')]:.3f} +/- {row[('PR_AUC', 'std')]:.3f} | "
            f"{row[('Brier', 'mean')]:.3f} +/- {row[('Brier', 'std')]:.3f} |")

    lines += ["", "Paired difference in ROC-AUC (KL + clinical minus KL only), "
                  "same folds:", ""]
    for model_name in MODELS:
        a = res[(res.model == model_name) & (res.features == "KL + clinical")]
        b = res[(res.model == model_name) & (res.features == "KL grade only")]
        d = a["ROC_AUC"].values - b["ROC_AUC"].values
        lines.append(f"- {model_name}: mean {d.mean():+.3f}, "
                     f"clinical model better in {(d > 0).sum()} of {len(d)} folds")
    lines += ["", "The +/- values are the spread across folds, not a confidence "
                  "interval."]

    report = "\n".join(lines)
    print("\n" + report)
    out = config.RESULTS_DIR / "kl_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()