"""
Experiment E5: SHAP Interpretability Analysis Before and After Leakage Removal.
Validates explainability shifts: proves that leakage-free pipeline forces the model to rely
on behavioral network flow features rather than lab artifact memorization.
Operationalizes Ahmed et al. (2024) and Zhang et al. (2026).
"""

import json
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

from config.config import RANDOM_SEEDS, RESULTS_DIR
from src.data.preprocessor import CICIDSPreprocessor
from src.data.leakage import LeakageAnalyzer
from src.models.pipeline import NIDSPipeline
from src.explainability.shap_analyzer import SHAPExplainerEngine

logger = logging.getLogger(__name__)


def run_experiment_e5(df_clean: pd.DataFrame, seed: int = RANDOM_SEEDS[0]) -> Dict[str, Any]:
    """
    Execute Experiment E5: Contrast SHAP explanations before vs after leakage removal.
    """
    logger.info("==================================================")
    logger.info("STARTING EXPERIMENT E5: SHAP EXPLAINABILITY ANALYSIS")
    logger.info("==================================================")

    # 1. SHAP on E1 Baseline (With Leakage)
    prep_e1 = CICIDSPreprocessor()
    prep_e1.feature_names = [c for c in df_clean.select_dtypes(include=[np.number]).columns
                             if c not in ["Binary_Label", "Label", "Clean_Label"]]
    X_train_e1, X_test_e1, y_train_e1, _ = prep_e1.split_data(
        df_clean, target_col="Binary_Label", random_state=seed
    )
    pipe_e1 = NIDSPipeline("random_forest", random_state=seed)
    pipe_e1.fit(X_train_e1, y_train_e1)

    explainer_e1 = SHAPExplainerEngine(pipe_e1.model, prep_e1.feature_names)
    X_test_scaled_e1 = pipe_e1.scaler.transform(X_test_e1)
    df_shap_e1 = explainer_e1.compute_shap_importance(X_test_scaled_e1, max_samples=150)

    # 2. SHAP on E2 Leakage-Free Pipeline
    leakage_analyzer = LeakageAnalyzer()
    df_leakage_free = leakage_analyzer.filter_leakage_features(df_clean)

    prep_e2 = CICIDSPreprocessor()
    prep_e2.feature_names = [c for c in df_leakage_free.select_dtypes(include=[np.number]).columns
                             if c not in ["Binary_Label", "Label", "Clean_Label"]]
    X_train_e2, X_test_e2, y_train_e2, _ = prep_e2.split_data(
        df_leakage_free, target_col="Binary_Label", random_state=seed
    )
    pipe_e2 = NIDSPipeline("random_forest", random_state=seed)
    pipe_e2.fit(X_train_e2, y_train_e2)

    explainer_e2 = SHAPExplainerEngine(pipe_e2.model, prep_e2.feature_names)
    X_test_scaled_e2 = pipe_e2.scaler.transform(X_test_e2)
    df_shap_e2 = explainer_e2.compute_shap_importance(X_test_scaled_e2, max_samples=150)

    # Generate comparison summary and dual-panel figure
    comparison = explainer_e2.generate_comparison_summary(df_shap_e1, df_shap_e2)

    output_path = RESULTS_DIR / "e5_shap_importance.json"
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"E5 SHAP comparison exported to {output_path}")
    return comparison
