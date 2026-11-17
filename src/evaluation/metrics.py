"""
Evaluation Metrics Engine for Network Intrusion Detection (CIC-IDS2017).
Computes imbalance-sensitive and cybersecurity-oriented metrics:
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa Coefficient
- Macro-F1 and Weighted-F1
- Per-class Recall and False Positive Rate (FPR)
- Operational throughput & latency (ms per 1,000 flows)"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef,
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from config.config import MITRE_ATTACK_MAPPING

logger = logging.getLogger(__name__)


class NIDSEvaluator:
    """Comprehensive evaluation module compliant with NIST AI RMF and NIST SP 800-94."""

    @staticmethod
    def calculate_binary_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        training_time: float = 0.0,
        inference_latency_ms_per_1k: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluate binary detection performance (0: BENIGN, 1: ATTACK).
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        # Imbalance-aware and cybersecurity metrics
        mcc = float(matthews_corrcoef(y_true, y_pred))
        kappa = float(cohen_kappa_score(y_true, y_pred))
        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))

        # False Positive Rate: FPR = FP / (FP + TN)
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        # Specificity: TN / (TN + FP)
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # Optional ROC-AUC
        auc = None
        if y_prob is not None:
            try:
                score_vector = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                auc = float(roc_auc_score(y_true, score_vector))
            except Exception as e:
                logger.warning(f"Failed to compute binary ROC-AUC: {e}")

        return {
            "mcc": mcc,
            "kappa": kappa,
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "false_positive_rate": fpr,
            "roc_auc": auc,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "training_time_s": float(training_time),
            "inference_latency_ms_per_1k": float(inference_latency_ms_per_1k)
        }

    @staticmethod
    def calculate_multiclass_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        y_prob: Optional[np.ndarray] = None,
        training_time: float = 0.0,
        inference_latency_ms_per_1k: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluate multiclass attack classification performance across all attack families.
        """
        mcc = float(matthews_corrcoef(y_true, y_pred))
        kappa = float(cohen_kappa_score(y_true, y_pred))
        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_true, y_pred)

        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        per_class_summary = []
        for idx, cls in enumerate(class_names):
            # Calculate per-class FPR from one-vs-rest confusion slice
            tp = cm[idx, idx]
            fn = np.sum(cm[idx, :]) - tp
            fp = np.sum(cm[:, idx]) - tp
            tn = np.sum(cm) - (tp + fn + fp)
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

            mitre_info = MITRE_ATTACK_MAPPING.get(cls, {})
            per_class_summary.append({
                "class": cls,
                "precision": float(per_class_precision[idx]),
                "recall": float(per_class_recall[idx]),
                "f1_score": float(per_class_f1[idx]),
                "false_positive_rate": fpr,
                "support": int(np.sum(cm[idx, :])),
                "mitre_technique": mitre_info.get("technique_id", "N/A"),
                "mitre_tactic": mitre_info.get("tactic", "N/A"),
                "severity": mitre_info.get("severity", "N/A")
            })

        return {
            "mcc": mcc,
            "kappa": kappa,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "accuracy": accuracy,
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
            "per_class_metrics": per_class_summary,
            "training_time_s": float(training_time),
            "inference_latency_ms_per_1k": float(inference_latency_ms_per_1k)
        }
