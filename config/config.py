import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure runtime directories exist
for path in [DATA_DIR, SAMPLE_DATA_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Default random seeds for deterministic evaluation (Arcos-Argudo et al., 2025)
RANDOM_SEEDS = [42, 101, 2024, 777, 999]
DEFAULT_SEED = 42

# Train / Test split configuration
TEST_SIZE = 0.20
STRATIFY = True

# Data Leakage Features (Gharghi & Dehghani, 2026; Arcos-Argudo et al., 2025)
# In CIC-IDS2017, these features allow models to memorize lab topology/scheduling
# rather than learning genuine malicious flow behavior:
# 1. 'Destination Port' / 'Dst Port': Memorizes victim service ports (e.g. 21, 22, 80).
# 2. 'Flow ID', 'Source IP', 'Destination IP': Memorizes fixed IP topology.
# 3. 'Timestamp': Memorizes attack schedule timeline (e.g., DoS on Wednesday).
# 4. 'Simulated / Artificial artifacts': Extreme outliers unique to flow exporters.
LEAKAGE_FEATURES = [
    "Destination Port",
    "Dst Port",
    "Flow ID",
    "Source IP",
    "Src IP",
    "Destination IP",
    "Dst IP",
    "Timestamp",
    "Unnamed: 0",
    "Index"
]

# Mapping raw CIC-IDS2017 labels to standardized categories
RAW_LABEL_MAPPING = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "Bot": "Botnet",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Web Attack \x96 Brute Force": "Web Attack",
    "Web Attack – Brute Force": "Web Attack",
    "Web Attack \x96 XSS": "Web Attack",
    "Web Attack – XSS": "Web Attack",
    "Web Attack \x96 Sql Injection": "Web Attack",
    "Web Attack – Sql Injection": "Web Attack"
}

# Rare attack classes (low sample count in CIC-IDS2017)
RARE_CLASSES = ["Heartbleed", "Infiltration", "Web Attack", "Botnet"]

# MITRE ATT&CK Mapping for Network Flows
MITRE_ATTACK_MAPPING = {
    "BENIGN": {
        "technique_id": "None",
        "technique_name": "Normal Authorized Traffic",
        "tactic": "Operational Normalcy",
        "severity": "Informational",
        "mitigation": "Continuous baseline monitoring"
    },
    "PortScan": {
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "tactic": "Discovery",
        "severity": "Medium",
        "mitigation": "Stateful firewall rate-limiting, disable unnecessary open ports"
    },
    "Brute Force": {
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "tactic": "Credential Access",
        "severity": "High",
        "mitigation": "Account lockout policies, multi-factor authentication (MFA), fail2ban"
    },
    "DoS": {
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "tactic": "Impact",
        "severity": "High",
        "mitigation": "Traffic shaping, anti-DoS appliances, upstream scrubbing centers"
    },
    "DDoS": {
        "technique_id": "T1498.001",
        "technique_name": "Direct Network Flood",
        "tactic": "Impact",
        "severity": "Critical",
        "mitigation": "Anycast routing, volumetric DDoS mitigation, ISP blackholing"
    },
    "Web Attack": {
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "tactic": "Initial Access",
        "severity": "Critical",
        "mitigation": "Web Application Firewall (WAF), parameterized queries, strict input validation"
    },
    "Botnet": {
        "technique_id": "T1071",
        "technique_name": "Application Layer Protocol: C2",
        "tactic": "Command and Control",
        "severity": "Critical",
        "mitigation": "DNS sinkholing, threat intelligence IP blocklists, outbound proxy egress inspection"
    },
    "Infiltration": {
        "technique_id": "T1210",
        "technique_name": "Exploitation of Remote Services",
        "tactic": "Lateral Movement",
        "severity": "Critical",
        "mitigation": "Network segmentation, micro-segmentation, rapid host patching"
    },
    "Heartbleed": {
        "technique_id": "T1212",
        "technique_name": "Exploitation for Credential Access",
        "tactic": "Credential Access",
        "severity": "Critical",
        "mitigation": "Upgrade OpenSSL >= 1.0.1g, revoke compromised TLS private keys"
    }
}
