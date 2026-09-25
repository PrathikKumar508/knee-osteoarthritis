"""
Data Preprocessing Pipeline Module
==================================
Builds reproducible, leakage-free scikit-learn preprocessing pipelines.
Ensures median imputation for numerical features, mode imputation and one-hot encoding
for categorical features, and feature scaling fitted ONLY on training splits.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import config


def build_preprocessing_pipeline(
    num_features: List[str],
    cat_features: List[str]
) -> ColumnTransformer:
    """
    Construct scikit-learn ColumnTransformer for numerical and categorical features.

    Numerical Pipeline:
      - SimpleImputer (median)
      - StandardScaler

    Categorical Pipeline:
      - SimpleImputer (most_frequent)
      - OneHotEncoder (handle_unknown='ignore', sparse_output=False)
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers = []
    if num_features:
        transformers.append(('num', num_pipeline, num_features))
    if cat_features:
        transformers.append(('cat', cat_pipeline, cat_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer, num_features: List[str], cat_features: List[str]) -> List[str]:
    """Retrieve feature names after preprocessing and One-Hot Encoding transformation."""
    feature_names = []
    
    if num_features and 'num' in preprocessor.named_transformers_:
        feature_names.extend(num_features)
        
    if cat_features and 'cat' in preprocessor.named_transformers_:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        encoded_cats = cat_encoder.get_feature_names_out(cat_features)
        feature_names.extend(encoded_cats)
        
    return feature_names
