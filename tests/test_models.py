"""
Unit tests for all 5 classifiers under unified training pipeline.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.preprocessor import CICIDSPreprocessor
from src.data.generator import generate_cicids_flow_sample
from src.models.pipeline import NIDSPipeline


@pytest.fixture(scope="module")
def sample_dataset():
    df_raw = generate_cicids_flow_sample(n_samples=400, random_seed=42)
    preprocessor = CICIDSPreprocessor()
    df_clean = preprocessor.clean_dataset(df_raw)
    return preprocessor.split_data(df_clean, target_col="Binary_Label", random_state=42)


@pytest.mark.parametrize("model_name", [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "svm"
])
def test_all_five_classifiers(model_name, sample_dataset):
    """Verify that all 5 required classifiers train and predict without errors."""
    X_train, X_test, y_train, y_test = sample_dataset

    pipeline = NIDSPipeline(classifier_name=model_name, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred, latency = pipeline.predict(X_test)
    assert len(y_pred) == len(X_test)
    assert latency > 0
    assert set(np.unique(y_pred)).issubset({0, 1})
