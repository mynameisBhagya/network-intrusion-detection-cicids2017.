"""
MITRE ATT&CK and NIST SP 800-94 Threat Intelligence Integration Module.
Maps classified network intrusion anomalies to adversary tactics, techniques, and triage actions.
Authors: Anannya Das (23000036) & Bhagya Sijumon (23000742)
British University in Dubai (BUiD)
"""

import logging
from typing import Dict, Any, Optional
from config.config import MITRE_ATTACK_MAPPING

logger = logging.getLogger(__name__)


class MitreAttackMapper:
    """Provides threat taxonomy and automated incident response guidance."""

    def __init__(self):
        self.mapping = MITRE_ATTACK_MAPPING

    def enrich_alert(self, attack_class: str, confidence: float = 1.0) -> Dict[str, Any]:
        """
        Enrich a detection alert with MITRE ATT&CK techniques, risk severity, and NIST mitigations.
        """
        info = self.mapping.get(attack_class, {
            "technique_id": "T1000",
            "technique_name": "Unknown Network Anomaly",
            "tactic": "Initial Access / Impact",
            "severity": "Medium",
            "description": f"Unrecognized attack signature flagged: {attack_class}",
            "mitigation": "Isolate affected host and conduct deep packet inspection."
        })

        alert = {
            "attack_class": attack_class,
            "confidence_pct": round(confidence * 100, 2),
            "mitre_technique_id": info.get("technique_id"),
            "mitre_technique_name": info.get("technique_name"),
            "mitre_tactic": info.get("tactic"),
            "severity": info.get("severity"),
            "recommended_mitigation": info.get("mitigation")
        }
        return alert

    def get_all_tactics_overview(self) -> Dict[str, Any]:
        """Return global matrix of monitored threat families."""
        return self.mapping
