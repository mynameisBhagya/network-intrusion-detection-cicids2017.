"""
Pipeline Orchestrator for Network Intrusion Detection (CIC-IDS2017).
Handles feature scaling, resampling (SMOTE / SMOTE-ENN), and training protocol
strictly on training splits to prevent data leakage and test set contamination.
Authors: Anannya Das (23000036) & Bhagya Sijumon (23000742)
British University in Dubai (BUiD)
"""

import time
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

from src.models.classifiers import get_classifier

logger = logging.getLogger(__name__)


class NIDSPipeline:
    """End-to-end training and inference pipeline for intrusion detection."""

    def __init__(
        self,
        classifier_name: str,
        resampling_strategy: Optional[str] = None,  # None, 'smote', 'smote_enn', 'class_weight'
        random_state: int = 42,
        is_multiclass: bool = False,
        **classifier_kwargs
    ):
        self.classifier_name = classifier_name
        self.resampling_strategy = resampling_strategy
        self.random_state = random_state
        self.is_multiclass = is_multiclass
        self.classifier_kwargs = classifier_kwargs

        self.scaler = StandardScaler()
        class_weight = "balanced" if resampling_strategy == "class_weight" else None
        self.model = get_classifier(
            classifier_name,
            class_weight=class_weight,
            random_state=random_state,
            is_multiclass=is_multiclass,
            **classifier_kwargs
        )

        self.training_time_seconds: float = 0.0
        self.feature_names: list = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit scaler and classifier strictly on the training partition.
        Applies synthetic resampling (SMOTE / SMOTE-ENN) ONLY to training data.
        """
        self.feature_names = list(X_train.columns)
        start_time = time.perf_counter()

        # Step 1: Fit and transform scaler on train partition
        X_scaled = self.scaler.fit_transform(X_train)

        # Step 2: Apply Resampling on training partition if requested
        if self.resampling_strategy == "smote":
            logger.info("Applying SMOTE oversampling strictly to training fold...")
            # For multi-class with rare classes, ensure k_neighbors does not exceed minority count
            min_class_count = y_train.value_counts().min()
            k_neighbors = min(5, max(1, min_class_count - 1))
            smote = SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
            X_scaled, y_train = smote.fit_resample(X_scaled, y_train)

        elif self.resampling_strategy == "smote_enn":
            logger.info("Applying hybrid SMOTE-ENN resampling strictly to training fold...")
            min_class_count = y_train.value_counts().min()
            k_neighbors = min(5, max(1, min_class_count - 1))
            smote_enn = SMOTEENN(smote=SMOTE(k_neighbors=k_neighbors, random_state=self.random_state),
                                 random_state=self.random_state)
            X_scaled, y_train = smote_enn.fit_resample(X_scaled, y_train)

        # Step 3: Train model
        logger.info(f"Training {self.classifier_name} on {len(X_scaled):,} samples...")
        self.model.fit(X_scaled, y_train)
        self.training_time_seconds = time.perf_counter() - start_time
        logger.info(f"Training finished in {self.training_time_seconds:.3f} seconds.")
        return self

    def predict(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Predict target classes and measure inference throughput (ms per 1,000 flows).
        """
        X_scaled = self.scaler.transform(X_test)
        start_time = time.perf_counter()
        y_pred = self.model.predict(X_scaled)
        inference_time = time.perf_counter() - start_time

        # Calculate latency in milliseconds per 1,000 flows
        flows_count = len(X_test)
        latency_ms_per_1k = (inference_time / max(1, flows_count)) * 1000 * 1000
        return y_pred, latency_ms_per_1k

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities if supported by the model."""
        X_scaled = self.scaler.transform(X_test)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        elif hasattr(self.model, "decision_function"):
            df_scores = self.model.decision_function(X_scaled)
            # Softmax / Sigmoid approximation for decision function
            if df_scores.ndim == 1:
                prob_pos = 1 / (1 + np.exp(-df_scores))
                return np.vstack([1 - prob_pos, prob_pos]).T
            else:
                exp_scores = np.exp(df_scores - np.max(df_scores, axis=1, keepdims=True))
                return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            raise NotImplementedError(f"{self.classifier_name} does not support probability prediction.")
