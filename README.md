# Predicting Knee Osteoarthritis Progression from Baseline Clinical Data

CSE mini project. The goal is to see how well information collected at a patient's first visit can predict whether their knee osteoarthritis (OA) gets worse over the following years. This is the first stage of a larger final-year project, where X-ray images will be added to the clinical data.

This is a student research prototype. It is not a medical tool and should not be used for diagnosis or treatment decisions.

## Current status

The code for the whole pipeline is written: target construction, preprocessing, patient-grouped splitting, model training, evaluation, SHAP plots and a Streamlit app.

I have not yet trained on real patient data. The `data/raw/` folder is empty because the OAI dataset requires registration and approval. When no real data is found, `src/train.py` generates a synthetic cohort of 1,000 patients and trains on that. The outcome in that cohort comes from a formula I wrote myself, so the results below only show that the pipeline runs end to end. They say nothing about real osteoarthritis.

Once the OAI files are in `data/raw/`, the same code runs on them and all results need to be regenerated.

## Research question

Do cheap baseline measurements (age, BMI, WOMAC pain, stiffness and disability scores, physical activity, injury and surgery history) predict future OA progression any better than the baseline Kellgren-Lawrence (KL) grade alone?

Many existing projects use the OAI data to grade or diagnose current OA from X-ray images. This project is about predicting the future from baseline information, and it asks specifically what the non-imaging variables add.

Planned analyses once real data is available (none of these are done yet):

1. Compare a model that uses only KL grade against the full model with all clinical features.
2. Repeat the analysis on early OA only (baseline KL 0 to 2).
3. Check calibration and performance separately for men and women.
4. Try a second definition of progression, such as an increase in WOMAC pain, and see whether the same features matter.

## Data and target

The dataset is the Osteoarthritis Initiative (OAI), a long-running NIH-funded study of knee OA. Access is through the NIMH Data Archive: https://nda.nih.gov/oai/

Files needed: a baseline clinical file (for example `AllClinical00.txt`) and KL grades at baseline and follow-up (for example `kxr_sq_bu00.txt` and `kxr_sq_bu06.txt`).

The target is binary. A knee counts as progressed (1) if its KL grade at follow-up is at least one grade higher than at baseline, otherwise 0. Knees with baseline KL 4 are excluded because they cannot get worse on this scale. Follow-up is set to visit 06 in `src/config.py`. These visit codes and column names are defaults and have to be checked against the OAI data dictionary once the real files are available.

## Avoiding data leakage

- Only baseline variables are used as inputs. Follow-up values are used only to build the target.
- The train/test split is grouped by patient ID, and the code asserts that no patient appears in both sets.
- Imputation, encoding and scaling are fitted on the training set only.

## Features
| Feature  | Description                              |
|----------|------------------------------------------|
| V00AGE   | Age in years                             |
| V00BMI   | Body mass index                          |
| V00WOMKP | WOMAC pain score                         |
| V00WOMAD | WOMAC disability score                   |
| V00WOMST | WOMAC stiffness score                    |
| V00PASE  | Physical Activity Scale for the Elderly  |
| V00SEX   | Sex                                      |
| V00KL    | Baseline KL grade                        |
| V00INJ   | Previous knee injury                     |
| V00SURG  | Previous knee surgery                    |

## Project structure

```text
knee-oa-progression/
│
├── data/
│   ├── raw/                  # Place raw OAI files here (.csv, .txt, .tsv, .sas7bdat)
│   └── processed/            # Joined & cleaned modeling datasets
│
├── notebooks/
│   └── 01_eda_and_baseline_model.ipynb   # EDA & baseline exploration
│
├── src/
│   ├── config.py             # Global pipeline settings & feature definitions
│   ├── inspect_dataset.py    # Stage 1: OAI file inspector & schema report generator
│   ├── data_preprocessing.py # Leakage-free sklearn ColumnTransformer pipeline
│   ├── feature_engineering.py# Target construction & feature extraction
│   ├── train.py              # Patient-grouped model training & pipeline serialization
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
│   └── results/              # Schema reports & markdown evaluation tables
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How to run

```
py -m pip install -r requirements.txt
py -m src.inspect_dataset
py -m src.train
py -m streamlit run app/app.py
```

`inspect_dataset` scans `data/raw/` and writes a report of the columns it finds. `train` uses real data if `data/processed/oai_processed.csv` exists, and the synthetic cohort otherwise.

## Current results (synthetic data)

Test split grouped by patient, decision threshold 0.5. These numbers come from `reports/results/evaluation_report.md`.

| Model               | ROC-AUC | Sensitivity | Specificity | Precision | F1     | Brier  |
|---------------------|---------|-------------|-------------|-----------|--------|--------|
| Logistic Regression | 0.7049  | 0.3733      | 0.8800      | 0.6512    | 0.4746 | 0.2075 |
| Random Forest       | 0.6401  | 0.1867      | 0.8800      | 0.4828    | 0.2692 | 0.2213 |
| XGBoost             | 0.6190  | 0.3733      | 0.8000      | 0.5283    | 0.4375 | 0.2313 |

Logistic Regression does best here mostly because the synthetic outcome was generated from a linear log-odds formula, so a linear model fits it naturally. This ordering may not hold on real data. Sensitivity is low at a 0.5 threshold, which shows that the choice of threshold matters if a model like this were ever used for screening.

## Limitations

- The current results come from synthetic data, so no conclusion about real OA can be drawn yet.
- Even with OAI data, everything comes from one cohort and has not been validated externally.
- Only one baseline time point is used, so changes over time are not modeled.
- SHAP values and feature importances describe how the model behaves. They do not show cause and effect.
- The risk tier cutoffs in the app (30% and 60% by default) are not clinically validated, and the monitoring suggestions in the app are examples only.
- OAI records are per knee. Grouping by patient stops leakage across the split, but two knees from the same patient are correlated and this needs to be handled properly with the real data.

## Future work

1. Run everything on real OAI data.
2. Add baseline knee X-rays and combine them with the clinical features using a CNN or vision transformer.
3. Model change over several visits instead of a single baseline.
4. Add Grad-CAM heatmaps for the images alongside the SHAP plots.

## Related work

OAI-based OA research is well established. Examples include DeepKnee, which grades OA from radiographs, and imedslab/OAProgression, which predicts progression from radiographs plus clinical data. This project is narrower. It uses clinical data only, controls for leakage carefully, and tests what those variables add beyond KL grade, as a base for the later multimodal version.