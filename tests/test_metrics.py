"""
Unit tests for imbalance-aware metrics and MITRE ATT&CK integration.
"""

import pytest
import numpy as np
from src.evaluation.metrics import NIDSEvaluator
from src.evaluation.mitre_mapper import MitreAttackMapper


def test_binary_metrics_calculation():
    """Verify calculation of MCC, Cohen's Kappa, Macro-F1, and FPR."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1])

    metrics = NIDSEvaluator.calculate_binary_metrics(y_true, y_pred)

    assert -1.0 <= metrics["mcc"] <= 1.0
    assert -1.0 <= metrics["kappa"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["false_positive_rate"] <= 1.0
    assert metrics["tp"] == 3
    assert metrics["fp"] == 1


def test_mitre_mapping():
    """Verify threat enrichment and MITRE ATT&CK technique lookups."""
    mapper = MitreAttackMapper()

    portscan_alert = mapper.enrich_alert("PortScan")
    assert portscan_alert["mitre_technique_id"] == "T1046"
    assert portscan_alert["mitre_tactic"] == "Discovery"

    dos_alert = mapper.enrich_alert("DoS")
    assert dos_alert["mitre_technique_id"] == "T1498"
    assert dos_alert["severity"] == "High"
