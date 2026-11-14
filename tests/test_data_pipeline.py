"""
Unit tests for data loader, cleaner, and preprocessor.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.preprocessor import CICIDSPreprocessor
from src.data.generator import generate_cicids_flow_sample


def test_generator_and_cleaning():
    """Verify data generation and sanitization pipeline."""
    df_raw = generate_cicids_flow_sample(n_samples=500, random_seed=42)
    assert len(df_raw) > 0
    assert "Label" in df_raw.columns

    # Introduce synthetic edge cases (inf, -inf, NaN, duplicate)
    df_dirty = df_raw.copy()
    df_dirty["Flow Duration"] = df_dirty["Flow Duration"].astype(float)
    df_dirty.loc[0, "Flow Duration"] = np.inf
    df_dirty.loc[1, "Flow Duration"] = -np.inf
    df_dirty.loc[2, "Flow Duration"] = np.nan
    df_dirty = pd.concat([df_dirty, df_dirty.iloc[[10]]], ignore_index=True)

    preprocessor = CICIDSPreprocessor()
    df_clean = preprocessor.clean_dataset(df_dirty)

    # Assert all infs and NaNs are removed
    assert not np.isinf(df_clean["Flow Duration"]).any()
    assert not df_clean["Flow Duration"].isna().any()
    assert "Binary_Label" in df_clean.columns
    assert "Clean_Label" in df_clean.columns


def test_stratified_split():
    """Verify that train-test split maintains exact class proportions."""
    df_raw = generate_cicids_flow_sample(n_samples=600, random_seed=42)
    preprocessor = CICIDSPreprocessor()
    df_clean = preprocessor.clean_dataset(df_raw)

    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df_clean, target_col="Binary_Label", test_size=0.2, random_state=42
    )

    assert len(X_train) + len(X_test) == len(df_clean)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    # Check stratification consistency within 5%
    assert abs(train_ratio - test_ratio) < 0.05
