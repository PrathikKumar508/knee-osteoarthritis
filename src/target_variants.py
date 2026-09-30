"""
Sensitivity analysis: how much do the results depend on how progression is
defined and on which knees are included?

For each variant, runs repeated patient-grouped cross-validation with Logistic
Regression and compares KL grade only vs. KL + clinical variables.

The main analysis (all knees, KL increase of at least 1) stays the primary
result. The variants are reported as sensitivity analyses, not used to pick a
more flattering target.

Run from the project root:
    python3 -m src.target_variants
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.config import config
from src.data_preprocessing import build_preprocessing_pipeline
from src.feature_engineering import create_progression_target

N_SPLITS = 5
N_REPEATS = 5
MIN_POSITIVES = 50

NUM = list(config.NUMERICAL_FEATURES)
CAT = list(config.CATEGORICAL_FEATURES)

FEATURE_SETS = {
    "KL only": ([], ["V00KL"]),
    "KL + clinical": (NUM, CAT),
}


def rise(d):
    return (d["V06KL"] - d["V00KL"]) >= 1


# name -> (which baseline KL grades to include, how progression is defined)
VARIANTS = {
    "Primary: baseline KL 0-3, KL rise >= 1":
        (lambda d: d["V00KL"].between(0, 3), rise),
    "Baseline KL 0-2 (early), KL rise >= 1":
        (lambda d: d["V00KL"].between(0, 2), rise),
    "Baseline KL 1-3, KL rise >= 1":
        (lambda d: d["V00KL"].between(1, 3), rise),
    "Baseline KL 2-3 (definite OA), KL rise >= 1":
        (lambda d: d["V00KL"].between(2, 3), rise),
    "Baseline KL 0-3, rise >= 1 and follow-up KL >= 2":
        (lambda d: d["V00KL"].between(0, 3), lambda d: rise(d) & (d["V06KL"] >= 2)),
}


def make_pipeline(num, cat):
    return Pipeline([
        ("prep", build_preprocessing_pipeline(num, cat)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def run_variant(df, y):
    groups = df[config.PATIENT_ID_COL].values
    rows = []
    for repeat in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=config.RANDOM_STATE + repeat)
        for fold, (tr, te) in enumerate(cv.split(df, y, groups)):
            for fs_name, (num, cat) in FEATURE_SETS.items():
                cols = num + cat
                pipe = make_pipeline(num, cat)
                pipe.fit(df.iloc[tr][cols], y[tr])
                p = pipe.predict_proba(df.iloc[te][cols])[:, 1]
                rows.append({"repeat": repeat, "fold": fold, "features": fs_name,
                             "ROC_AUC": roc_auc_score(y[te], p),
                             "PR_AUC": average_precision_score(y[te], p)})
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv(config.PROCESSED_DATA_DIR / "oai_processed.csv")
    # Keeps knees with both KL grades and baseline KL < 4
    base = create_progression_target(raw)

    lines = [
        "# Sensitivity analysis: target definition and knee subgroup",
        "",
        f"Logistic Regression, {N_REPEATS} x {N_SPLITS}-fold patient-grouped cross-validation. "
        "AUC values from different subgroups are not directly comparable because the "
        "populations differ. Look at the KL + clinical minus KL-only difference within each row.",
        "",
        "| Variant | Knees | Patients | Progression rate | ROC-AUC KL only | ROC-AUC KL + clinical | Difference | PR-AUC KL only | PR-AUC KL + clinical | Folds better |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for name, (include, define) in VARIANTS.items():
        d = base[include(base)].reset_index(drop=True)
        y = define(d).astype(int).values
        n_pos = int(y.sum())
        print(f"\n{name}: {len(d):,} knees, {n_pos} progressors ({y.mean():.3f})")
        if n_pos < MIN_POSITIVES or n_pos > len(y) - MIN_POSITIVES:
            print("  skipped (too few events in one class)")
            lines.append(f"| {name} | {len(d):,} | - | {y.mean():.3f} | skipped: too few events | | | | | |")
            continue

        res = run_variant(d, y)
        a = res[res.features == "KL only"].sort_values(["repeat", "fold"])
        b = res[res.features == "KL + clinical"].sort_values(["repeat", "fold"])
        diff = b["ROC_AUC"].values - a["ROC_AUC"].values
        line = (f"| {name} | {len(d):,} | {d[config.PATIENT_ID_COL].nunique():,} | {y.mean():.3f} | "
                f"{a['ROC_AUC'].mean():.3f} | {b['ROC_AUC'].mean():.3f} | {diff.mean():+.3f} | "
                f"{a['PR_AUC'].mean():.3f} | {b['PR_AUC'].mean():.3f} | "
                f"{(diff > 0).sum()}/{len(diff)} |")
        print("  " + line)
        lines.append(line)

    lines += ["",
              "Only the primary definition was chosen in advance. The other rows show "
              "how sensitive the conclusion is to that choice."]
    out = config.RESULTS_DIR / "target_variants.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()