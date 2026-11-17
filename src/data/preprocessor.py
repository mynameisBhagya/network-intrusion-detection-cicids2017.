import logging
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config.config import RAW_LABEL_MAPPING, RANDOM_SEEDS, TEST_SIZE

logger = logging.getLogger(__name__)


class CICIDSPreprocessor:
    """End-to-end preprocessing and audit engine for CIC-IDS2017 flows."""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.constant_columns: List[str] = []
        self.feature_names: List[str] = []
        self.cleaning_stats: Dict[str, Any] = {}

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute deterministic data sanitization:
        - Strip string columns
        - Convert infinite values to NaN
        - Drop null / NaN instances
        - Remove duplicate flows
        - Drop zero-variance (constant) features
        """
        initial_rows = len(df)
        logger.info(f"Initiating dataset cleaning on {initial_rows:,} records...")

        df = df.copy()

        # Handle Label standardization
        if "Label" in df.columns:
            df["Label"] = df["Label"].astype(str).str.strip()
            # Map granular labels to unified attack classes
            df["Clean_Label"] = df["Label"].map(lambda x: RAW_LABEL_MAPPING.get(x, x))
            df["Binary_Label"] = (df["Clean_Label"] != "BENIGN").astype(int)
        else:
            raise KeyError("Dataset is missing mandatory 'Label' column.")

        # Identify numeric columns for Inf/NaN checks
        non_feature_cols = ["Label", "Clean_Label", "Binary_Label"]
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in non_feature_cols]

        # Replace +/- infinity with NaN
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Drop NaN values
        nan_dropped_rows = df[numeric_cols].isna().any(axis=1).sum()
        df = df.dropna(subset=numeric_cols)

        # Drop duplicates
        dup_rows = df.duplicated(subset=numeric_cols).sum()
        df = df.drop_duplicates(subset=numeric_cols)

        # Identify and drop constant (zero-variance) columns
        variances = df[numeric_cols].var()
        self.constant_columns = variances[variances == 0].index.tolist()
        if self.constant_columns:
            logger.info(f"Pruning {len(self.constant_columns)} constant features: {self.constant_columns}")
            df = df.drop(columns=self.constant_columns)
            numeric_cols = [c for c in numeric_cols if c not in self.constant_columns]

        final_rows = len(df)
        self.feature_names = numeric_cols

        self.cleaning_stats = {
            "initial_records": initial_rows,
            "nan_or_inf_dropped": int(nan_dropped_rows),
            "duplicates_dropped": int(dup_rows),
            "constant_columns_dropped": self.constant_columns,
            "final_records": final_rows,
            "total_features": len(self.feature_names)
        }
        logger.info(f"Cleaning complete: {final_rows:,} clean flows preserved (retained {final_rows/initial_rows*100:.2f}%).")
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str = "Binary_Label",
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_SEEDS[0]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Perform stratified train-test split before any feature scaling or resampling,
        ensuring complete isolation of evaluation data (preventing test set contamination).
        """
        features = [c for c in self.feature_names if c in df.columns]
        X = df[features]
        y = df[target_col]

        # Fit label encoder if multi-class string target
        if y.dtype == object or isinstance(y.iloc[0], str):
            y = pd.Series(self.label_encoder.fit_transform(y), index=y.index, name=target_col)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        logger.info(f"Stratified split executed (seed={random_state}): "
                    f"Train={len(X_train):,}, Test={len(X_test):,}")
        return X_train, X_test, y_train, y_test
