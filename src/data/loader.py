"""
Data Loader for CIC-IDS2017 Dataset.
Supports both Parquet and CSV formats with automated column normalization and memory optimization.
Authors: Anannya Das (23000036) & Bhagya Sijumon (23000742)
British University in Dubai (BUiD)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CICIDSDataLoader:
    """Robust data loader for CIC-IDS2017 dataset."""

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self.data_path = Path(data_path) if data_path else None

    @staticmethod
    def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing spaces and special characters from column names."""
        df = df.copy()
        df.columns = [col.strip() for col in df.columns]
        # Standardize target label column if variant casing exists
        rename_dict = {}
        for col in df.columns:
            if col.lower() == "label":
                rename_dict[col] = "Label"
        if rename_dict:
            df = df.rename(columns=rename_dict)
        return df

    def load_data(
        self,
        filepath: Optional[Union[str, Path]] = None,
        sample_frac: Optional[float] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Load CIC-IDS2017 data from Parquet or CSV.
        
        Args:
            filepath: Path to parquet or csv file. If None, uses self.data_path.
            sample_frac: Optional float fraction to sample (e.g. 0.1 for 10%).
            random_state: Random state for deterministic sampling.
            
        Returns:
            pd.DataFrame: Cleaned raw dataframe with normalized headers.
        """
        target_path = Path(filepath) if filepath else self.data_path
        if target_path is None or not target_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {target_path}")

        logger.info(f"Loading CIC-IDS2017 dataset from: {target_path}")
        
        if target_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(target_path)
        elif target_path.suffix.lower() == ".csv":
            df = pd.read_csv(target_path, low_memory=False)
        else:
            raise ValueError(f"Unsupported file format '{target_path.suffix}'. Use .parquet or .csv")

        df = self.normalize_column_names(df)
        logger.info(f"Loaded {df.shape[0]:,} flows across {df.shape[1]} attributes.")

        if sample_frac and 0.0 < sample_frac < 1.0:
            logger.info(f"Subsampling {sample_frac*100:.1f}% of data with seed={random_state}...")
            if "Label" in df.columns:
                df = df.groupby("Label", group_keys=False).apply(
                    lambda x: x.sample(frac=sample_frac, random_state=random_state)
                ).reset_index(drop=True)
            else:
                df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
            logger.info(f"Subsampled shape: {df.shape[0]:,} flows.")

        return df
