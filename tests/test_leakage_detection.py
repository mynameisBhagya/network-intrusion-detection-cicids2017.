"""
Unit tests for data leakage detection and ablation engine.
"""

import pytest
import pandas as pd
from src.data.leakage import LeakageAnalyzer
from src.data.generator import generate_cicids_flow_sample


def test_leakage_audit_and_filtering():
    """Verify that Destination Port and other leakage vectors are properly flagged and purged."""
    df_raw = generate_cicids_flow_sample(n_samples=200, random_seed=42)

    analyzer = LeakageAnalyzer()
    audit_report = analyzer.audit_features(df_raw)

    assert "Destination Port" in audit_report["explicit_leakage_found"]

    df_filtered = analyzer.filter_leakage_features(df_raw)
    assert "Destination Port" not in df_filtered.columns

    # Verify behavioral flow features remain intact
    assert "Flow Duration" in df_filtered.columns
    assert "Flow IAT Mean" in df_filtered.columns
    assert "Fwd Packet Length Mean" in df_filtered.columns
