"""
Configuration Module for Knee OA Progression Risk Prediction
============================================================
Central configuration defining file paths, OAI identifier column names,
baseline clinical features, target progression thresholds, and model parameters.
All parameters can be dynamically overridden after Stage 1 dataset inspection.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class OAIConfig:
    # Directory Paths
    RAW_DATA_DIR: Path = Path("data/raw")
    PROCESSED_DATA_DIR: Path = Path("data/processed")
    MODELS_DIR: Path = Path("models")
    REPORTS_DIR: Path = Path("reports")
    FIGURES_DIR: Path = Path("reports/figures")
    RESULTS_DIR: Path = Path("reports/results")

    # Patient & Visit Identifiers (Default OAI Standard Names)
    PATIENT_ID_COL: str = "ID"
    SIDE_COL: str = "SIDE"  # 'RIGHT' or 'LEFT' knee if multi-knee record
    VISIT_COL: str = "VISIT"
    BASELINE_VISIT_VAL: str = "00"  # Baseline visit code (0-Month)
    FOLLOWUP_VISIT_VAL: str = "06"  # Follow-up visit code (e.g., 48-Month)

    # Target Definition Settings
    BASELINE_KL_COL: str = "V00KL"   # Baseline Kellgren-Lawrence Grade (0-4)
    FOLLOWUP_KL_COL: str = "V06KL"   # Follow-up Kellgren-Lawrence Grade (0-4)
    TARGET_COL: str = "oa_progression"
    PROGRESSION_THRESHOLD: int = 1   # Delta KL >= 1 indicates radiographic OA progression

    # Standard Candidate Baseline Features (Inspected/Mapped dynamically)
    NUMERICAL_FEATURES: List[str] = field(default_factory=lambda: [
        "V00AGE",      # Patient Age
        "V00BMI",      # Body Mass Index
        "V00WOMKP",    # Baseline WOMAC Pain Score
        "V00WOMAD",    # Baseline WOMAC Disability Score
        "V00WOMST",    # Baseline WOMAC Stiffness Score
        "V00PASE",     # Physical Activity Scale for the Elderly
    ])

    CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda: [
        "V00SEX",      # Sex (1=Male, 2=Female)
        "V00KL",       # Baseline KL Grade (0, 1, 2, 3)
        "V00INJ",      # History of Knee Injury (0=No, 1=Yes)
        "V00SURG",     # History of Knee Surgery (0=No, 1=Yes)
    ])

    # Model Parameters & Training Settings
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.20
    VAL_SIZE: float = 0.15
    GROUP_COL: str = "ID"  # Group column for GroupShuffleSplit (avoids patient leakage)

    # Risk Classification Thresholds
    LOW_RISK_THRESHOLD: float = 0.30
    HIGH_RISK_THRESHOLD: float = 0.60

    def get_feature_list(self) -> List[str]:
        """Return combined list of numerical and categorical baseline input features."""
        return self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES


# Global config instance
config = OAIConfig()
