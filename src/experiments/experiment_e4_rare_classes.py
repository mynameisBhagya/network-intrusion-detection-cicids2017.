"""
Experiment E4: Rare Attack Class Evaluation and Grouping Strategy.
Investigates detection performance on rare attack classes (Heartbleed, Infiltration, Web Attacks, Botnet).
Operationalizes Aseel & Keyhanipour (2026) and Bulavas et al. (2021).
"""

import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from config.config import RANDOM_SEEDS, RESULTS_DIR, RARE_CLASSES
from src.data.preprocessor import CICIDSPreprocessor
from src.data.leakage import LeakageAnalyzer
from src.models.pipeline import NIDSPipeline
from src.evaluation.metrics import NIDSEvaluator

logger = logging.getLogger(__name__)


def run_experiment_e4(df_clean: pd.DataFrame, seed: int = RANDOM_SEEDS[0]) -> Dict[str, Any]:
    """
    Execute Experiment E4: Rare class impact analysis.
    """
    logger.info("==================================================")
    logger.info("STARTING EXPERIMENT E4: RARE ATTACK CLASS ANALYSIS")
    logger.info("==================================================")

    leakage_analyzer = LeakageAnalyzer()
    df_leakage_free = leakage_analyzer.filter_leakage_features(df_clean)

    # Strategy A: Raw Multiclass (All classes retained)
    preprocessor_a = CICIDSPreprocessor()
    preprocessor_a.feature_names = [c for c in df_leakage_free.select_dtypes(include=[np.number]).columns
                                    if c not in ["Binary_Label", "Label", "Clean_Label", "Grouped_Label"]]

    class_names_raw = sorted(df_leakage_free["Clean_Label"].unique().tolist())
    X_train_a, X_test_a, y_train_a, y_test_a = preprocessor_a.split_data(
        df_leakage_free, target_col="Clean_Label", random_state=seed
    )

    pipeline_raw = NIDSPipeline("xgboost", random_state=seed, is_multiclass=True)
    pipeline_raw.fit(X_train_a, y_train_a)
    y_pred_raw, lat_raw = pipeline_raw.predict(X_test_a)

    eval_raw = NIDSEvaluator.calculate_multiclass_metrics(
        y_test_a.values, y_pred_raw, class_names=class_names_raw,
        training_time=pipeline_raw.training_time_seconds,
        inference_latency_ms_per_1k=lat_raw
    )

    # Strategy B: Merging rare attacks into "Advanced / Rare Threats" category
    df_grouped = df_leakage_free.copy()
    df_grouped["Grouped_Label"] = df_grouped["Clean_Label"].map(
        lambda x: "Rare Threat Group" if x in RARE_CLASSES else x
    )
    class_names_grouped = sorted(df_grouped["Grouped_Label"].unique().tolist())

    preprocessor_b = CICIDSPreprocessor()
    preprocessor_b.feature_names = preprocessor_a.feature_names

    X_train_b, X_test_b, y_train_b, y_test_b = preprocessor_b.split_data(
        df_grouped, target_col="Grouped_Label", random_state=seed
    )
    pipeline_grouped = NIDSPipeline("xgboost", random_state=seed, is_multiclass=True)
    pipeline_grouped.fit(X_train_b, y_train_b)
    y_pred_grouped, lat_grouped = pipeline_grouped.predict(X_test_b)

    eval_grouped = NIDSEvaluator.calculate_multiclass_metrics(
        y_test_b.values, y_pred_grouped, class_names=class_names_grouped,
        training_time=pipeline_grouped.training_time_seconds,
        inference_latency_ms_per_1k=lat_grouped
    )

    results = {
        "raw_classes_strategy": {
            "mcc": eval_raw["mcc"],
            "macro_f1": eval_raw["macro_f1"],
            "rare_classes_performance": [p for p in eval_raw["per_class_metrics"] if p["class"] in RARE_CLASSES]
        },
        "grouped_classes_strategy": {
            "mcc": eval_grouped["mcc"],
            "macro_f1": eval_grouped["macro_f1"],
            "per_class_metrics": eval_grouped["per_class_metrics"]
        }
    }

    output_path = RESULTS_DIR / "e4_rare_classes_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"E4 results exported to {output_path}")
    return results
