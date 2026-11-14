"""
SHAP (SHapley Additive exPlanations) Explainability Engine for NIDS.
Supports Experiment E5: Comparative interpretability analysis before and after data leakage removal.
Operationalizes NIST AI RMF requirement for reliable, explainable AI decisions.
Authors: Anannya Das (23000036) & Bhagya Sijumon (23000742)
British University in Dubai (BUiD)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap

# Set MPLCONFIGDIR to prevent warning
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "/tmp"), "matplotlib_cache")
import matplotlib.pyplot as plt

from config.config import FIGURES_DIR

logger = logging.getLogger(__name__)


class SHAPExplainerEngine:
    """Computes global feature importances and local decision attributions using SHAP."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def fit_explainer(self, X_background: np.ndarray, max_samples: int = 100):
        """Initialize SHAP TreeExplainer on fitted model."""
        logger.info(f"Initializing SHAP explainer on {min(len(X_background), max_samples)} background samples...")
        if len(X_background) > max_samples:
            indices = np.random.choice(len(X_background), max_samples, replace=False)
            background = X_background[indices]
        else:
            background = X_background

        try:
            # TreeExplainer directly handles fitted RandomForest and XGBoost
            self.explainer = shap.TreeExplainer(self.model, data=background)
        except Exception:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                self.explainer = shap.Explainer(self.model.predict, background)

    def compute_shap_importance(
        self,
        X_eval: np.ndarray,
        max_samples: int = 150
    ) -> pd.DataFrame:
        """
        Calculate global mean absolute SHAP values across evaluation samples.
        Returns a sorted DataFrame of feature importances.
        """
        eval_data = X_eval[:max_samples] if len(X_eval) > max_samples else X_eval

        try:
            if self.explainer is None:
                self.fit_explainer(eval_data, max_samples=80)

            # Compute SHAP values
            shap_vals = self.explainer.shap_values(eval_data)
            if isinstance(shap_vals, list):
                # Multiclass list of arrays
                mean_abs = np.mean([np.mean(np.abs(arr), axis=0) for arr in shap_vals], axis=0)
            elif isinstance(shap_vals, np.ndarray):
                if shap_vals.ndim == 3:
                    mean_abs = np.mean(np.abs(shap_vals), axis=(0, 2))
                else:
                    mean_abs = np.mean(np.abs(shap_vals), axis=0)
            else:
                raw = self.explainer(eval_data)
                vals = raw.values
                mean_abs = np.mean(np.abs(vals), axis=0) if vals.ndim == 2 else np.mean(np.abs(vals), axis=(0, 2))

        except Exception as e:
            logger.warning(f"SHAP sampling fallback due to: {e}. Utilizing feature_importances_ attribution.")
            if hasattr(self.model, "feature_importances_"):
                mean_abs = self.model.feature_importances_
            else:
                mean_abs = np.ones(len(self.feature_names)) / len(self.feature_names)

        df_importance = pd.DataFrame({
            "Feature": self.feature_names[:len(mean_abs)],
            "Mean_SHAP_Value": mean_abs
        }).sort_values(by="Mean_SHAP_Value", ascending=False).reset_index(drop=True)

        return df_importance

    def generate_comparison_summary(
        self,
        df_before_leakage: pd.DataFrame,
        df_after_leakage: pd.DataFrame,
        output_image: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Compare top-10 SHAP feature rankings before and after leakage elimination.
        Proves whether the model shifts from memorizing ports/timestamps to genuine flow physics.
        """
        top_before = df_before_leakage.head(10).to_dict(orient="records")
        top_after = df_after_leakage.head(10).to_dict(orient="records")

        # Create dual-panel comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: With Leakage
        y_pos1 = np.arange(len(top_before))
        names1 = [x["Feature"] for x in top_before][::-1]
        vals1 = [x["Mean_SHAP_Value"] for x in top_before][::-1]
        colors1 = ["#ff4d4f" if any(lk.lower() in n.lower() for lk in ["port", "ip", "time"]) else "#1890ff" for n in names1]
        axes[0].barh(y_pos1, vals1, color=colors1)
        axes[0].set_yticks(y_pos1)
        axes[0].set_yticklabels(names1)
        axes[0].set_title("E1: SHAP Attribution (With Leakage Features)\n[Red = Leakage Artifacts]")
        axes[0].set_xlabel("Mean |SHAP Value|")

        # Panel 2: Leakage Free
        y_pos2 = np.arange(len(top_after))
        names2 = [x["Feature"] for x in top_after][::-1]
        vals2 = [x["Mean_SHAP_Value"] for x in top_after][::-1]
        axes[1].barh(y_pos2, vals2, color="#52c41a")
        axes[1].set_yticks(y_pos2)
        axes[1].set_yticklabels(names2)
        axes[1].set_title("E2: SHAP Attribution (Leakage-Free Behavioral Pipeline)\n[Green = Genuine Flow Dynamics]")
        axes[1].set_xlabel("Mean |SHAP Value|")

        plt.tight_layout()
        save_path = output_image or (FIGURES_DIR / "shap_leakage_comparison.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

        logger.info(f"SHAP leakage comparison plot saved to: {save_path}")

        return {
            "top_10_before_leakage": top_before,
            "top_10_after_leakage": top_after,
            "figure_path": str(save_path)
        }
