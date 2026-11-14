"""
Experiment E2: Data Leakage Elimination & Performance Ablation.
Reruns evaluation with known leakage features stripped (Destination Port, IP/time artifacts).
Quantifies empirical score inflation caused by artifact memorization.
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


def run_experiment_e2(
    df_clean: pd.DataFrame,
    e1_results: Dict[str, Any],
    seeds: List[int] = None
) -> Dict[str, Any]:
    """
    Execute Experiment E2: Leakage-free behavioral pipeline and inflation calculation.
    """
    seeds = seeds or RANDOM_SEEDS[:3]
    models = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "svm"]
    results = {"binary": {}, "inflation_analysis": {}}

    # Strip leakage columns
    leakage_analyzer = LeakageAnalyzer()
    df_leakage_free = leakage_analyzer.filter_leakage_features(df_clean)

    preprocessor = CICIDSPreprocessor()
    preprocessor.feature_names = [c for c in df_leakage_free.select_dtypes(include=[np.number]).columns
                                  if c not in ["Binary_Label", "Label", "Clean_Label"]]

    logger.info("==================================================")
    logger.info("STARTING EXPERIMENT E2: LEAKAGE ELIMINATION")
    logger.info(f"Retained {len(preprocessor.feature_names)} purely behavioral flow attributes.")
    logger.info("==================================================")

    for model_name in models:
        seed_metrics = []
        for seed in seeds:
            X_train, X_test, y_train, y_test = preprocessor.split_data(
                df_leakage_free, target_col="Binary_Label", random_state=seed
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

        results["binary"][model_name] = {
            "mcc_mean": avg_mcc,
            "mcc_std": float(np.std([m["mcc"] for m in seed_metrics])),
            "kappa_mean": avg_kappa,
            "macro_f1_mean": avg_f1,
            "accuracy_mean": avg_acc,
            "fpr_mean": avg_fpr,
            "seed_runs": seed_metrics
        }

        # Calculate Inflation from E1
        if "binary" in e1_results and model_name in e1_results["binary"]:
            e1_mcc = e1_results["binary"][model_name]["mcc_mean"]
            e1_acc = e1_results["binary"][model_name]["accuracy_mean"]
            inflation_mcc = e1_mcc - avg_mcc
            inflation_acc = e1_acc - avg_acc
            results["inflation_analysis"][model_name] = {
                "e1_baseline_mcc": e1_mcc,
                "e2_leakage_free_mcc": avg_mcc,
                "mcc_drop_due_to_leakage_removal": inflation_mcc,
                "e1_baseline_acc": e1_acc,
                "e2_leakage_free_acc": avg_acc,
                "acc_drop_due_to_leakage_removal": inflation_acc
            }
            logger.info(
                f"E2 [{model_name}]: MCC={avg_mcc:.4f} (Baseline={e1_mcc:.4f}, Delta={inflation_mcc:+.4f}) | "
                f"Acc={avg_acc:.4f} (Baseline={e1_acc:.4f}, Delta={inflation_acc:+.4f})"
            )

    output_path = RESULTS_DIR / "e2_leakage_ablation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"E2 results exported to {output_path}")
    return results
