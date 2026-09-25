"""
Feature Engineering & Target Definition Module
===============================================
Constructs the longitudinal Knee OA progression target variable from baseline and
follow-up measurements while strictly preventing feature data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from src.config import config, OAIConfig


def create_progression_target(
    df: pd.DataFrame,
    baseline_kl_col: str = config.BASELINE_KL_COL,
    followup_kl_col: str = config.FOLLOWUP_KL_COL,
    target_col: str = config.TARGET_COL,
    threshold: int = config.PROGRESSION_THRESHOLD
) -> pd.DataFrame:
    """
    Construct binary progression target Y:
    Y = 1 if (Followup KL Grade - Baseline KL Grade >= threshold)
    Y = 0 otherwise.

    Patients with baseline KL Grade = 4 (end-stage OA) are excluded as progression 
    cannot be radiographically measured beyond grade 4.
    """
    df = df.copy()

    if baseline_kl_col not in df.columns or followup_kl_col not in df.columns:
        raise ValueError(
            f"Missing required KL columns for target construction: '{baseline_kl_col}' or '{followup_kl_col}'"
        )

    # Convert to numeric
    df[baseline_kl_col] = pd.to_numeric(df[baseline_kl_col], errors='coerce')
    df[followup_kl_col] = pd.to_numeric(df[followup_kl_col], errors='coerce')

    # Exclude invalid KL grades or baseline KL = 4 (max severe grade)
    valid_mask = df[baseline_kl_col].notnull() & df[followup_kl_col].notnull() & (df[baseline_kl_col] < 4)
    df = df[valid_mask].copy()

    # Calculate change in KL grade
    kl_diff = df[followup_kl_col] - df[baseline_kl_col]
    
    # Target: 1 if progression >= threshold (e.g. 1 grade increase), else 0
    df[target_col] = (kl_diff >= threshold).astype(int)

    print(f"Target Construction Summary:")
    print(f" - Valid Cohort Records: {len(df):,}")
    print(f" - Progression Cases (Y=1): {(df[target_col] == 1).sum():,} ({df[target_col].mean()*100:.2f}%)")
    print(f" - Non-Progression Cases (Y=0): {(df[target_col] == 0).sum():,} ({(1-df[target_col].mean())*100:.2f}%)")

    return df


def enforce_baseline_features_only(
    df: pd.DataFrame,
    input_features: List[str],
    target_col: str = config.TARGET_COL,
    patient_id_col: str = config.PATIENT_ID_COL
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Strict Leakage Prevention: Extract strictly baseline input features X, 
    target Y, and patient group IDs for splitting.
    Removes all follow-up columns from X.
    """
    missing_cols = [c for c in input_features if c not in df.columns]
    if missing_cols:
        print(f"[!] Warning: The following candidate features were not found in dataset: {missing_cols}")

    available_features = [c for c in input_features if c in df.columns]
    
    X = df[available_features].copy()
    y = df[target_col].copy()
    groups = df[patient_id_col].copy() if patient_id_col in df.columns else pd.Series(df.index)

    print(f"\nExtracted ML Datasets:")
    print(f" - Features X shape: {X.shape}")
    print(f" - Target y shape: {y.shape}")
    print(f" - Unique Patients: {groups.nunique():,}")

    return X, y, groups


def save_processed_dataset(df: pd.DataFrame, output_path: Path = config.PROCESSED_DATA_DIR / "oai_processed.csv"):
    """Save processed modeling dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed dataset to: {output_path}")
