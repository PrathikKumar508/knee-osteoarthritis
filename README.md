# Predicting Knee Osteoarthritis Progression from Baseline Clinical Data

CSE mini project. The goal is to see how well information collected at a patient's first visit can predict whether their knee osteoarthritis (OA) gets worse over the following four years, and whether cheap clinical measurements add anything beyond the baseline X-ray grade. This is the first stage of a larger final-year project, where X-ray images will be added to the clinical data.

This is a student research prototype. It is not a medical tool and should not be used for diagnosis or treatment decisions.

## Research question

Do baseline clinical variables (age, sex, BMI, WOMAC pain, disability and stiffness scores, physical activity, knee injury and surgery history) predict future OA progression better than the baseline Kellgren-Lawrence (KL) grade alone?

## Summary of findings

- Baseline data predict progression only modestly. With repeated patient-grouped cross-validation, the best models reach a ROC-AUC of about 0.65 (0.5 is chance).
- KL grade alone and the clinical variables alone perform about the same (ROC-AUC about 0.62 for Logistic Regression).
- Combining them helps a little. ROC-AUC rises by about 0.03 and PR-AUC from 0.18 to 0.22, and this held in every cross-validation fold. The folds reuse the same patients, so this is a consistent effect, not a formal significance test.
- The probabilities are only slightly more informative than always predicting the overall progression rate (Brier score 0.116 vs 0.120). The models are not accurate enough to guide decisions about individual patients.

## Data

Dataset: the Osteoarthritis Initiative (OAI), a long-running NIH-funded study of knee OA. Access is through the NIMH Data Archive: https://nda.nih.gov/oai/

Files used (pipe-delimited ASCII, placed in `data/raw/`):

- `AllClinical00.txt`: baseline age, BMI, PASE, WOMAC scores, injury and surgery history
- `Enrollees.txt`: sex
- `AllClinical06.txt`: used only to check that visit 06 is about 4 years after baseline (mean age difference 4.03 years)
- `KXR_SQ_BU00.txt` and `KXR_SQ_BU06.txt` (from the X-ray image assessments): KL grade at baseline and at visit 06 (48 months)

### Building the modeling table

`src/build_dataset.py` turns these files into one table with one row per knee:

1. Clinical variables are read from the wide files, where right and left knee values are in separate columns, and reshaped to one row per knee with a `SIDE` column.
2. KL grades are read from the X-ray files (one row per knee, `SIDE` coded as `1: Right` and `2: Left`) and matched on patient ID and side.
3. The OAI stores many values as text such as `2: Grade 2` or `1: Male`. The leading number is extracted, and missing-value codes become blanks.
4. Some knees have several readings from different reading projects (15, 37 and 43). Project 15 covers almost every knee, so it is used when it has a grade. Only about 2% of knees have readings that disagree.
5. Knees without a KL grade at both time points are dropped.

Counts: 4,796 participants, 9,592 knees, 7,151 knees with both KL grades, 6,986 after excluding baseline KL 4 (progression cannot be measured beyond grade 4). These knees belong to 3,659 patients. Missing values in the features are 1% or less.

### Target

A knee counts as progressed (1) if its KL grade at 48 months is at least one grade higher than at baseline, otherwise 0. 974 of 6,986 knees (13.9%) progressed.

### Features

Model variable | OAI source       | Description
---------------|------------------|-----------------------------------------------
 V00AGE        | V00AGE           | Age in years
 V00SEX        | P02SEX           | Sex (1 male, 2 female)
 V00BMI        | P01BMI           | Body mass index
 V00PASE       | V00PASE          | Physical Activity Scale for the Elderly
 V00WOMKP      | V00WOMKPR / L    | WOMAC pain, per knee
 V00WOMAD      | V00WOMADLR / L   | WOMAC disability, per knee
 V00WOMST      | V00WOMSTFR / L   | WOMAC stiffness, per knee
 V00INJ        | P01INJR / L      | Previous injury of that knee
 V00SURG       | P01KSURGR / L    | Previous surgery on that knee
 V00KL         | KXR_SQ_BU00      | Baseline KL grade

## Avoiding data leakage

- Only baseline variables are used as inputs. Follow-up values are used only to build the target.
- Both knees of a patient always stay in the same training or test set (grouped by patient ID), and the training code checks that no patient appears in both.
- Imputation, encoding and scaling are fitted on the training data only.

## Methods

Two models are compared: Logistic Regression and a small Random Forest (200 trees, depth 4). XGBoost is supported by `src/train.py` when the package is installed, but it is not part of the results below.

The main comparison (`src/compare_models.py`) uses 5 repeats of 5-fold stratified cross-validation, grouped by patient, and three feature sets: KL grade only, clinical variables without KL, and KL plus clinical variables. The metrics are ROC-AUC, PR-AUC (average precision) and Brier score. These do not depend on a decision threshold, which matters here because only 14% of knees progress.

## Results

Cross-validated results (mean +/- spread across folds). Reference values for a useless model: ROC-AUC 0.5, PR-AUC 0.139 (the progression rate), Brier 0.120.

Model                 | Features                  | ROC-AUC       | PR-AUC        | Brier
----------------------|---------------------------|---------------|---------------|--------------
Logistic Regression   | KL grade only             | 0.618 ± 0.023 | 0.184 ± 0.013 | 0.117 ± 0.001
Logistic Regression   | Clinical only (no KL)     | 0.618 ± 0.021 | 0.202 ± 0.019 | 0.118 ± 0.001
Logistic Regression   | KL + clinical             | 0.649 ± 0.023 | 0.224 ± 0.024 | 0.116 ± 0.002
Random Forest         | KL grade only             | 0.618 ± 0.023 | 0.184 ± 0.013 | 0.117 ± 0.001
Random Forest         | Clinical only (no KL)     | 0.608 ± 0.020 | 0.197 ± 0.019 | 0.118 ± 0.001
Random Forest         | KL + clinical             | 0.644 ± 0.022 | 0.220 ± 0.022 | 0.116 ± 0.001

Paired difference in ROC-AUC (KL + clinical minus KL only) on the same folds: +0.031 for Logistic Regression and +0.026 for Random Forest, positive in all 25 folds. The spread values are not confidence intervals. The KL-only rows are identical for both models because a single categorical feature gives both models the same ranking of knees.

The single train/test split in `src/train.py` gives similar discrimination (ROC-AUC 0.631 for Logistic Regression and 0.617 for Random Forest). At the default 0.5 threshold neither model flags any knee as a progressor (sensitivity 0), because no predicted probability reaches 0.5 when the base rate is 14%. Threshold-based metrics are therefore not reported as results.

The full comparison table is saved in `reports/results/kl_comparison.md`.

## Project structure

```text
knee-oa-progression/
│
├── data/
│   ├── raw/                  # Raw OAI files (.txt), including the X-ray assessment folders
│   └── processed/            # oai_processed.csv (one row per knee)
│
├── notebooks/
│   └── 01_eda_and_baseline_model.ipynb   # EDA & baseline exploration
│
├── src/
│   ├── config.py             # Global pipeline settings & feature definitions
│   ├── inspect_dataset.py    # Stage 1: OAI file inspector & schema report generator
│   ├── build_dataset.py      # Builds the one-row-per-knee modeling table
│   ├── data_preprocessing.py # Leakage-free sklearn ColumnTransformer pipeline
│   ├── feature_engineering.py# Target construction & feature extraction
│   ├── train.py              # Patient-grouped model training & pipeline serialization
│   ├── compare_models.py     # KL-only vs clinical features, repeated grouped CV
│   ├── evaluate.py           # Multi-metric evaluation
│   ├── explainability.py     # SHAP feature importance analysis
│   └── predict.py            # Prediction & risk tier inference
│
├── models/
│   └── trained_pipeline.pkl  # Serialized model pipeline artifact
│
├── app/
│   └── app.py                # Streamlit web application
│
├── reports/
│   ├── figures/              # Saved ROC curves, PR curves & SHAP plots
│   └── results/              # Schema reports, evaluation and comparison tables
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How to run

```
python3 -m pip install -r requirements.txt
python3 -m src.inspect_dataset     # profile the raw files
python3 -m src.build_dataset       # build data/processed/oai_processed.csv
python3 -m src.train               # train models, save the pipeline and plots
python3 -m src.compare_models      # KL-only vs clinical comparison
python3 -m streamlit run app/app.py
```

On Windows, use `py` instead of `python3`. The raw OAI data is not included in the repository and must be downloaded separately.

## Limitations

- Performance is modest (ROC-AUC about 0.65), so the model is not useful for individual clinical decisions.
- The target is noisy. A change from KL 0 to 1 involves the "doubtful" grade, which even expert readers disagree on, so part of the 14% progression rate may be reader variability and not real disease change.
- Only participants who have both X-rays are included, so people who dropped out, died or missed the 48-month visit are missing. The results may not apply to them.
- The OAI combines subcohorts recruited in different ways (progression, incidence and non-exposed control). They are pooled here and the analysis does not adjust for cohort.
- The rule that prefers reading project 15, and the meaning of the injury and surgery variables (`P01INJ*`, `P01KSURG*`), come from column names and row counts. They should be confirmed against the OAI documentation.
- Only one baseline time point is used, so changes over time are not modeled, and there is no external validation.
- SHAP values and feature importances describe how the model behaves. They do not show cause and effect.
- The app's risk tier cutoffs (30% and 60% by default) were set before real data was available. With a 14% base rate they would label almost everyone low risk, and they are not clinically validated. Some app content (the threshold simulator, monitoring suggestions) is illustrative and not derived from the model.
- Two knees from the same patient are correlated. Grouping by patient prevents leakage, but the knees are still counted as separate rows.

## Future work

1. Recalibrate the app's risk tiers using the real distribution of predicted risks, and replace the simulated threshold curve with one computed from real predictions.
2. Add a patient-level bootstrap for confidence intervals on the AUC differences, and include XGBoost in the comparison.
3. Add baseline knee X-rays and combine them with the clinical features using a CNN or vision transformer.
4. Model change over several visits instead of a single baseline.
5. Add Grad-CAM heatmaps for the images alongside the SHAP plots.

## Related work

OAI-based OA research is well established. Examples include DeepKnee, which grades OA from radiographs, and imedslab/OAProgression, which predicts progression from radiographs plus clinical data. This project is narrower. It uses clinical data only, controls for leakage carefully, and tests what those variables add beyond KL grade, as a base for the later multimodal version.