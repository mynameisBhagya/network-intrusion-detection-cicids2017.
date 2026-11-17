
import logging
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

from config.config import LEAKAGE_FEATURES

logger = logging.getLogger(__name__)


class LeakageAnalyzer:
    """Detects, inspects, and isolates data leakage artifacts in network flow records."""

    def __init__(self, suspect_features: List[str] = None):
        self.suspect_features = suspect_features or LEAKAGE_FEATURES
        self.detected_leakage: List[str] = []

    def audit_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Inspect dataset columns for known leakage vectors:
        - Port identifiers (e.g. Destination Port)
        - Host addressing (Source/Destination IP)
        - Session timestamps and sequential row IDs
        """
        existing_cols = df.columns.tolist()
        self.detected_leakage = [col for col in existing_cols if col in self.suspect_features]

        port_suspects = [c for c in existing_cols if "port" in c.lower()]
        ip_suspects = [c for c in existing_cols if "ip" in c.lower() or "address" in c.lower()]
        time_suspects = [c for c in existing_cols if "time" in c.lower() or "date" in c.lower()]

        audit_report = {
            "explicit_leakage_found": self.detected_leakage,
            "port_related_columns": port_suspects,
            "ip_related_columns": ip_suspects,
            "time_related_columns": time_suspects,
            "recommendation": "Strip all non-behavioral network identifiers to ensure realistic generalizability."
        }
        logger.info(f"Leakage audit detected {len(self.detected_leakage)} explicit leakage vectors: {self.detected_leakage}")
        return audit_report

    def filter_leakage_features(self, df: pd.DataFrame, extra_excludes: List[str] = None) -> pd.DataFrame:
        """
        Remove all leakage-prone and non-causal identifier columns,
        preserving strictly physical behavioral traffic features (durations, packet sizes, inter-arrival times, TCP flags).
        """
        excludes = set(self.suspect_features)
        if extra_excludes:
            excludes.update(extra_excludes)

        drop_targets = [col for col in df.columns if col in excludes]
        logger.info(f"Purging {len(drop_targets)} leakage columns from feature space: {drop_targets}")
        return df.drop(columns=drop_targets, errors="ignore")

    @staticmethod
    def calculate_leakage_correlation(df: pd.DataFrame, target_col: str = "Binary_Label") -> pd.DataFrame:
        """
        Compute Pearson correlation of features against the target label
        to quantify synthetic bias and leakage potency.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col not in numeric_df.columns:
            logger.warning(f"Target column '{target_col}' not found in numeric features.")
            return pd.DataFrame()

        corrs = numeric_df.corrwith(numeric_df[target_col]).abs().sort_values(ascending=False)
        corr_df = pd.DataFrame({
            "Feature": corrs.index,
            "Absolute_Correlation": corrs.values
        }).query("Feature != @target_col")
        return corr_df.reset_index(drop=True)
