"""
Experiment E1: Baseline Classification with All Features (Including Potential Leakage).
Evaluates all 5 classifiers (LR, DT, RF, XGBoost, SVM) in Binary and Multiclass settings
across fixed random seeds.
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
from src.models.pipeline import NIDSPipeline
from src.evaluation.metrics import NIDSEvaluator

logger = logging.getLogger(__name__)


def run_experiment_e1(df_clean: pd.DataFrame, seeds: List[int] = None) -> Dict[str, Any]:
    """
    Execute Experiment E1: Baseline training on unpruned feature space.
    """
    seeds = seeds or RANDOM_SEEDS[:3]
    models = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "svm"]
    results = {"binary": {}, "multiclass": {}}

    preprocessor = CICIDSPreprocessor()
    # Keep all clean features including port
    preprocessor.feature_names = [c for c in df_clean.select_dtypes(include=[np.number]).columns
                                  if c not in ["Binary_Label", "Label", "Clean_Label"]]

    logger.info("==================================================")
    logger.info("STARTING EXPERIMENT E1: BASELINE (ALL FEATURES)")
    logger.info("==================================================")

    # 1. Binary Classification
    for model_name in models:
        seed_metrics = []
        for seed in seeds:
            X_train, X_test, y_train, y_test = preprocessor.split_data(
                df_clean, target_col="Binary_Label", random_state=seed
            )
            pipeline = NIDSPipeline(classifier_name=model_name, random_state=seed, is_multiclass=False)
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
        avg_kappa = float(np.mean([m["kappa"] for m in seed_metrics]))
        avg_f1 = float(np.mean([m["macro_f1"] for m in seed_metrics]))
        avg_acc = float(np.mean([m["accuracy"] for m in seed_metrics]))
        avg_fpr = float(np.mean([m["false_positive_rate"] for m in seed_metrics]))
        avg_latency = float(np.mean([m["inference_latency_ms_per_1k"] for m in seed_metrics]))

        results["binary"][model_name] = {
            "mcc_mean": avg_mcc,
            "mcc_std": float(np.std([m["mcc"] for m in seed_metrics])),
            "kappa_mean": avg_kappa,
            "macro_f1_mean": avg_f1,
            "accuracy_mean": avg_acc,
            "fpr_mean": avg_fpr,
            "latency_ms_mean": avg_latency,
            "seed_runs": seed_metrics
        }
        logger.info(f"E1 Binary [{model_name}]: MCC={avg_mcc:.4f}, Kappa={avg_kappa:.4f}, Macro-F1={avg_f1:.4f}")

    # 2. Multiclass Classification (Top 2 representative ensembles: RF & XGBoost)
    class_names = sorted(df_clean["Clean_Label"].unique().tolist())
    for model_name in ["random_forest", "xgboost"]:
        seed_metrics = []
        for seed in seeds:
            X_train, X_test, y_train, y_test = preprocessor.split_data(
                df_clean, target_col="Clean_Label", random_state=seed
            )
            pipeline = NIDSPipeline(classifier_name=model_name, random_state=seed, is_multiclass=True)
            pipeline.fit(X_train, y_train)
            y_pred, latency = pipeline.predict(X_test)

            eval_res = NIDSEvaluator.calculate_multiclass_metrics(
                y_test.values, y_pred, class_names=class_names,
                training_time=pipeline.training_time_seconds,
                inference_latency_ms_per_1k=latency
            )
            seed_metrics.append(eval_res)

        results["multiclass"][model_name] = {
            "mcc_mean": float(np.mean([m["mcc"] for m in seed_metrics])),
            "kappa_mean": float(np.mean([m["kappa"] for m in seed_metrics])),
            "macro_f1_mean": float(np.mean([m["macro_f1"] for m in seed_metrics])),
            "accuracy_mean": float(np.mean([m["accuracy"] for m in seed_metrics])),
            "per_class_metrics": seed_metrics[0]["per_class_metrics"]
        }

    output_path = RESULTS_DIR / "e1_baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"E1 results exported to {output_path}")
    return results
