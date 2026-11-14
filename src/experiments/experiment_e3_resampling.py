"""
Experiment E3: Class Imbalance Remedies & Resampling Comparison.
Compares None, Class Weighting, SMOTE, and SMOTE-ENN strictly fit on training splits.
Operationalizes Singh et al. (2025) and Hemmatian et al. (2025).
Authors: Anannya Das (23000036) & Bhagya Sijumon (23000742)
British University in Dubai (BUiD)
"""

import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from config.config import RANDOM_SEEDS, RESULTS_DIR
from src.data.preprocessor import CICIDSPreprocessor
from src.data.leakage import LeakageAnalyzer
from src.models.pipeline import NIDSPipeline
from src.evaluation.metrics import NIDSEvaluator

logger = logging.getLogger(__name__)


def run_experiment_e3(df_clean: pd.DataFrame, seeds: List[int] = None) -> Dict[str, Any]:
    """
    Execute Experiment E3: Resampling strategy comparison on Random Forest and XGBoost.
    """
    seeds = seeds or RANDOM_SEEDS[:3]
    strategies = ["none", "class_weight", "smote", "smote_enn"]
    eval_models = ["random_forest", "xgboost"]
    results = {}

    leakage_analyzer = LeakageAnalyzer()
    df_leakage_free = leakage_analyzer.filter_leakage_features(df_clean)

    preprocessor = CICIDSPreprocessor()
    preprocessor.feature_names = [c for c in df_leakage_free.select_dtypes(include=[np.number]).columns
                                  if c not in ["Binary_Label", "Label", "Clean_Label"]]

    logger.info("==================================================")
    logger.info("STARTING EXPERIMENT E3: CLASS IMBALANCE RESAMPLING")
    logger.info("==================================================")

    for strat in strategies:
        results[strat] = {}
        for model_name in eval_models:
            seed_metrics = []
            for seed in seeds:
                X_train, X_test, y_train, y_test = preprocessor.split_data(
                    df_leakage_free, target_col="Binary_Label", random_state=seed
                )
                pipeline = NIDSPipeline(
                    classifier_name=model_name,
                    resampling_strategy=strat if strat != "none" else None,
                    random_state=seed,
                    is_multiclass=False
                )
                pipeline.fit(X_train, y_train)
                y_pred, latency = pipeline.predict(X_test)
                try:
                    y_prob = pipeline.predict_proba(X_test)
                except Exception:
                    y_prob = None

                eval_res = NIDSEvaluator.calculate_binary_metrics(
                    y_test.values, y_pred, y_prob,
                    training_time=pipeline.training_time_seconds,
                    inference_latency_ms_per_1k=latency
                )
                seed_metrics.append(eval_res)

            avg_mcc = float(np.mean([m["mcc"] for m in seed_metrics]))
            avg_recall = float(np.mean([m["recall"] for m in seed_metrics]))
            avg_fpr = float(np.mean([m["false_positive_rate"] for m in seed_metrics]))
            avg_f1 = float(np.mean([m["macro_f1"] for m in seed_metrics]))

            results[strat][model_name] = {
                "mcc_mean": avg_mcc,
                "recall_mean": avg_recall,
                "fpr_mean": avg_fpr,
                "macro_f1_mean": avg_f1
            }
            logger.info(f"E3 [{strat.upper()} | {model_name}]: MCC={avg_mcc:.4f}, Attack Recall={avg_recall:.4f}, FPR={avg_fpr:.4f}")

    output_path = RESULTS_DIR / "e3_resampling_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"E3 results exported to {output_path}")
    return results
