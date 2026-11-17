"""
Statistical Significance Engine for Classifier Comparison.
Implements paired Wilcoxon signed-rank tests and paired t-tests across 5 fixed random seeds
to validate experimental claims with statistical rigor (p < 0.05).
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class StatisticalComparator:
    """Performs rigorous statistical comparison of ML models evaluated over multi-seed runs."""

    @staticmethod
    def compare_models(
        scores_model_a: List[float],
        scores_model_b: List[float],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
        metric_name: str = "MCC"
    ) -> Dict[str, Any]:
        """
        Execute paired Wilcoxon signed-rank test and paired Student's t-test.
        
        Args:
            scores_model_a: List of scores across seeds for Model A
            scores_model_b: List of scores across seeds for Model B
            model_a_name: Display label for Model A
            model_b_name: Display label for Model B
            metric_name: Name of evaluated metric (e.g. 'MCC', 'Macro-F1')

        Returns:
            Dict containing mean, std, p-values, and statistical conclusion
        """
        a = np.array(scores_model_a)
        b = np.array(scores_model_b)

        mean_a, std_a = float(np.mean(a)), float(np.std(a))
        mean_b, std_b = float(np.mean(b)), float(np.std(b))
        diff = a - b

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(a, b)

        # Wilcoxon signed-rank test (non-parametric)
        # If all differences are zero, p-value is 1.0
        if np.all(diff == 0):
            w_stat, w_pval = 0.0, 1.0
        else:
            try:
                w_stat, w_pval = stats.wilcoxon(a, b)
            except Exception as e:
                logger.warning(f"Wilcoxon calculation fallback: {e}")
                w_stat, w_pval = float(t_stat), float(t_pval)

        is_significant = bool(w_pval < 0.05)
        superior_model = model_a_name if mean_a > mean_b else model_b_name

        report = {
            "metric": metric_name,
            "model_a": {"name": model_a_name, "mean": mean_a, "std": std_a, "scores": scores_model_a},
            "model_b": {"name": model_b_name, "mean": mean_b, "std": std_b, "scores": scores_model_b},
            "difference_mean": float(np.mean(diff)),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_pvalue": float(w_pval),
            "paired_ttest_stat": float(t_stat),
            "paired_ttest_pvalue": float(t_pval),
            "statistically_significant_at_0.05": is_significant,
            "conclusion": (
                f"Difference between {model_a_name} and {model_b_name} is statistically significant "
                f"(p={w_pval:.4f} < 0.05). {superior_model} performs superiorly."
                if is_significant else
                f"No statistically significant difference observed (p={w_pval:.4f} >= 0.05)."
            )
        }
        return report
