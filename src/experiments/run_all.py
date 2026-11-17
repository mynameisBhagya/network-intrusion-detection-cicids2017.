"""
Master Pipeline Orchestration Script for CIC-IDS2017 ML Benchmark.
Executes Experiments E1 through E5, conducts multi-seed statistical significance testing,
persists trained model artifacts, and prepares hydrated results for the Streamlit dashboard.
"""

import sys
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import SAMPLE_DATA_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_SEEDS
from src.data.loader import CICIDSDataLoader
from src.data.preprocessor import CICIDSPreprocessor
from src.data.leakage import LeakageAnalyzer
from src.data.generator import generate_cicids_flow_sample
from src.experiments.experiment_e1_baseline import run_experiment_e1
from src.experiments.experiment_e2_leakage import run_experiment_e2
from src.experiments.experiment_e3_resampling import run_experiment_e3
from src.experiments.experiment_e4_rare_classes import run_experiment_e4
from src.experiments.experiment_e5_shap import run_experiment_e5
from src.evaluation.statistical import StatisticalComparator
from src.models.pipeline import NIDSPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("================================================================================")
    logger.info("EXPLAINABLE LEAKAGE-FREE NIDS BENCHMARK SUITE (CIC-IDS2017)")
    logger.info("British University in Dubai | CSCY413 Special Topics in Cybersecurity")
    logger.info("Authors: Anannya Das (23000036) & Bhagya Sijumon (23000742)")
    logger.info("================================================================================")

    # 1. Ensure Dataset Exists
    sample_parquet = SAMPLE_DATA_DIR / "cicids2017_sample.parquet"
    if not sample_parquet.exists():
        logger.info("Sample dataset not found on disk. Generating calibrated benchmark data...")
        generate_cicids_flow_sample(n_samples=12000, random_seed=RANDOM_SEEDS[0])

    # 2. Load and Clean
    loader = CICIDSDataLoader(sample_parquet)
    df_raw = loader.load_data()
    preprocessor = CICIDSPreprocessor()
    df_clean = preprocessor.clean_dataset(df_raw)

    # 3. Execute Experiment E1: Baseline
    e1_results = run_experiment_e1(df_clean, seeds=RANDOM_SEEDS[:3])

    # 4. Execute Experiment E2: Leakage Elimination
    e2_results = run_experiment_e2(df_clean, e1_results=e1_results, seeds=RANDOM_SEEDS[:3])

    # 5. Execute Experiment E3: Resampling Comparison
    e3_results = run_experiment_e3(df_clean, seeds=RANDOM_SEEDS[:3])

    # 6. Execute Experiment E4: Rare Class Impact
    e4_results = run_experiment_e4(df_clean, seed=RANDOM_SEEDS[0])

    # 7. Execute Experiment E5: SHAP Interpretability Before vs After Leakage
    e5_results = run_experiment_e5(df_clean, seed=RANDOM_SEEDS[0])

    # 8. Multi-Seed Statistical Significance Comparison (Random Forest vs XGBoost)
    rf_mccs = [r["mcc"] for r in e2_results["binary"]["random_forest"]["seed_runs"]]
    xgb_mccs = [r["mcc"] for r in e2_results["binary"]["xgboost"]["seed_runs"]]
    stat_report = StatisticalComparator.compare_models(
        rf_mccs, xgb_mccs, "Random Forest", "XGBoost", metric_name="MCC"
    )
    stat_file = RESULTS_DIR / "statistical_tests.json"
    import json
    with open(stat_file, "w") as f:
        json.dump(stat_report, f, indent=2)
    logger.info(f"Statistical significance tests exported to {stat_file}")
    logger.info(f"Wilcoxon test conclusion: {stat_report['conclusion']}")

    # 9. Train and Persist Production-Ready Leakage-Free Models for Triage Dashboard
    logger.info("Persisting best trained models for real-time dashboard triage...")
    leakage_analyzer = LeakageAnalyzer()
    df_leakage_free = leakage_analyzer.filter_leakage_features(df_clean)

    # A. Binary Model (Random Forest)
    prep_prod_bin = CICIDSPreprocessor()
    prep_prod_bin.feature_names = [c for c in df_leakage_free.select_dtypes(include=[np.number]).columns
                                   if c not in ["Binary_Label", "Label", "Clean_Label"]]
    X_train_bin, _, y_train_bin, _ = prep_prod_bin.split_data(
        df_leakage_free, target_col="Binary_Label", random_state=RANDOM_SEEDS[0]
    )
    pipe_bin = NIDSPipeline("random_forest", random_state=RANDOM_SEEDS[0])
    pipe_bin.fit(X_train_bin, y_train_bin)

    with open(MODELS_DIR / "rf_leakage_free_binary.pkl", "wb") as f:
        pickle.dump(pipe_bin, f)

    # B. Multiclass Model (XGBoost)
    prep_prod_multi = CICIDSPreprocessor()
    prep_prod_multi.feature_names = prep_prod_bin.feature_names
    X_train_multi, _, y_train_multi, _ = prep_prod_multi.split_data(
        df_leakage_free, target_col="Clean_Label", random_state=RANDOM_SEEDS[0]
    )
    pipe_multi = NIDSPipeline("xgboost", random_state=RANDOM_SEEDS[0], is_multiclass=True)
    pipe_multi.fit(X_train_multi, y_train_multi)

    with open(MODELS_DIR / "xgb_leakage_free_multiclass.pkl", "wb") as f:
        pickle.dump(pipe_multi, f)

    with open(MODELS_DIR / "label_encoders.pkl", "wb") as f:
        pickle.dump({
            "classes": prep_prod_multi.label_encoder.classes_,
            "feature_names": prep_prod_multi.feature_names
        }, f)

    logger.info("================================================================================")
    logger.info("ALL BENCHMARK EXPERIMENTS COMPLETED SUCCESSFULLY!")
    logger.info("Trained model artifacts and JSON results are ready for Streamlit dashboard.")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
