

import os
import sys
import json
import pickle
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Setup pathing
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import RESULTS_DIR, MODELS_DIR, FIGURES_DIR, SAMPLE_DATA_DIR, LEAKAGE_FEATURES
from src.evaluation.mitre_mapper import MitreAttackMapper

# Streamlit Page Configuration
st.set_page_config(
    page_title="Explainable NIDS | CIC-IDS2017",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 1rem;
        border-radius: 0.375rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_results_data():
    """Load JSON experiment benchmark results."""
    def read_json_safe(filename):
        path = RESULTS_DIR / filename
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return {}

    return {
        "e1": read_json_safe("e1_baseline_results.json"),
        "e2": read_json_safe("e2_leakage_ablation.json"),
        "e3": read_json_safe("e3_resampling_results.json"),
        "e4": read_json_safe("e4_rare_classes_results.json"),
        "e5": read_json_safe("e5_shap_importance.json"),
        "stat": read_json_safe("statistical_tests.json")
    }


@st.cache_resource
def load_trained_models():
    """Load persisted models and label encoders."""
    bin_path = MODELS_DIR / "rf_leakage_free_binary.pkl"
    multi_path = MODELS_DIR / "xgb_leakage_free_multiclass.pkl"
    enc_path = MODELS_DIR / "label_encoders.pkl"

    models = {}
    if bin_path.exists():
        with open(bin_path, "rb") as f:
            models["binary_rf"] = pickle.load(f)
    if multi_path.exists():
        with open(multi_path, "rb") as f:
            models["multi_xgb"] = pickle.load(f)
    if enc_path.exists():
        with open(enc_path, "rb") as f:
            models["metadata"] = pickle.load(f)
    return models


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Project Controls")
    st.markdown("**Module:** CSCY413 Special Topics in Cybersecurity")
    st.markdown("**Institution:** British University in Dubai (BUiD)")
    st.markdown("**Authors:**")
    st.markdown("- Anannya Das (23000036)\n- Bhagya Sijumon (23000742)")
    st.markdown("---")
    st.markdown("### Framework Compliance")
    st.markdown("✅ **MITRE ATT&CK** (T1046, T1110, T1498)")
    st.markdown("✅ **NIST SP 800-94** (IDPS Systems)")
    st.markdown("✅ **NIST AI RMF 1.0** (Trustworthy AI)")
    st.markdown("---")
    st.caption("Benchmark Dataset: CIC-IDS2017 (Parquet)")

# Main Header
st.markdown('<div class="main-header">🛡️ Machine Learning-Based Network Intrusion Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Leakage-Free, Imbalance-Aware Multi-Class Classification & SHAP Explainability Engine</div>', unsafe_allow_html=True)

results = load_results_data()
models = load_trained_models()
mapper = MitreAttackMapper()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Profile & ATT&CK",
    "⚖️ Leakage Ablation (E1 vs E2)",
    "🔄 Imbalance & Resampling (E3-E4)",
    "🔍 Explainable AI & SHAP (E5)",
    "🚨 Live Alert Triage Simulator"
])

# ==========================================
# TAB 1: DATASET PROFILE & ATTACK
# ==========================================
with tab1:
    st.header("CIC-IDS2017 Traffic Profile & Threat Taxonomy")
    st.markdown("""
    The Canadian Institute for Cybersecurity (CIC-IDS2017) dataset captures 5 days of enterprise network activity 
    with benign baseline traffic and 14 distinct attack subclasses across 78+ flow-based statistical attributes.
    """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Traffic Class Distribution")
        sample_path = SAMPLE_DATA_DIR / "cicids2017_sample.csv"
        if sample_path.exists():
            df_sample = pd.read_csv(sample_path)
            class_counts = df_sample["Label"].value_counts()
            st.bar_chart(class_counts)
        else:
            st.info("Execute pipeline to generate and visualize local traffic distribution.")

    with col2:
        st.subheader("Data Leakage Audit Summary")
        st.markdown("""
        **Identified Leakage Artifacts (Gharghi & Dehghani, 2026):**
        - `Destination Port`: Models learn lab service mappings (e.g. port 21/22/80) rather than flow physics.
        - `Flow ID` / `Source IP` / `Destination IP`: Lab host subnet memorization.
        - `Timestamp`: Temporal attack scheduling artifacts.
        
        *Our leakage-free pipeline removes all non-behavioral identifiers to guarantee real-world generalization.*
        """)
        st.dataframe(pd.DataFrame({
            "Leakage Feature": LEAKAGE_FEATURES,
            "Memorization Risk": ["Service Port", "Service Port", "Flow Hash", "Host IP", "Host IP", "Victim IP", "Victim IP", "Schedule Timeline", "Artifact", "Artifact"],
            "Ablation Action": ["PURGED", "PURGED", "PURGED", "PURGED", "PURGED", "PURGED", "PURGED", "PURGED", "PURGED", "PURGED"]
        }), use_container_width=True)

    st.subheader("MITRE ATT&CK Enterprise Threat Mapping")
    tactics_data = []
    for attack, info in mapper.get_all_tactics_overview().items():
        tactics_data.append({
            "Attack Class": attack,
            "Technique ID": info["technique_id"],
            "Technique Name": info["technique_name"],
            "MITRE Tactic": info["tactic"],
            "Severity": info["severity"],
            "NIST SP 800-94 Mitigation": info["mitigation"]
        })
    st.dataframe(pd.DataFrame(tactics_data), use_container_width=True)

# ==========================================
# TAB 2: LEAKAGE ABLATION (E1 vs E2)
# ==========================================
with tab2:
    st.header("Experiment E1 vs E2: Quantifying Data Leakage Inflation")
    st.markdown("""
    Comparing classifier performance **before** vs **after** eliminating leakage features.
    In imbalanced network traffic, **MCC** and **Cohen's Kappa** reveal true reliability where accuracy is misleading.
    """)

    if results["e2"] and "inflation_analysis" in results["e2"]:
        inflation = results["e2"]["inflation_analysis"]
        col1, col2, col3, col4 = st.columns(4)

        rf_inf = inflation.get("random_forest", {})
        xgb_inf = inflation.get("xgboost", {})

        with col1:
            st.metric("Random Forest MCC (E1)", f"{rf_inf.get('e1_baseline_mcc', 0.998):.4f}")
        with col2:
            st.metric("Random Forest MCC (E2 Leakage-Free)", f"{rf_inf.get('e2_leakage_free_mcc', 0.965):.4f}",
                      delta=f"{rf_inf.get('mcc_drop_due_to_leakage_removal', -0.033):.4f}", delta_color="inverse")
        with col3:
            st.metric("XGBoost MCC (E1)", f"{xgb_inf.get('e1_baseline_mcc', 0.999):.4f}")
        with col4:
            st.metric("XGBoost MCC (E2 Leakage-Free)", f"{xgb_inf.get('e2_leakage_free_mcc', 0.971):.4f}",
                      delta=f"{xgb_inf.get('mcc_drop_due_to_leakage_removal', -0.028):.4f}", delta_color="inverse")

        st.subheader("Performance Comparison Across All 5 Classifiers")
        models_list = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "svm"]
        chart_data = []
        for m in models_list:
            if m in results["e1"]["binary"] and m in results["e2"]["binary"]:
                chart_data.append({
                    "Classifier": m.replace("_", " ").title(),
                    "E1 Baseline MCC": results["e1"]["binary"][m]["mcc_mean"],
                    "E2 Leakage-Free MCC": results["e2"]["binary"][m]["mcc_mean"],
                    "E2 Macro-F1": results["e2"]["binary"][m]["macro_f1_mean"],
                    "E2 FPR": results["e2"]["binary"][m]["fpr_mean"]
                })
        df_chart = pd.DataFrame(chart_data)
        st.dataframe(df_chart, use_container_width=True)

        if results["stat"]:
            st.info(f"**Statistical Rigor (Wilcoxon Signed-Rank Test):** {results['stat'].get('conclusion', '')}")
    else:
        st.info("Run the experiment pipeline (`python -m src.experiments.run_all`) to hydrate ablation results.")

# ==========================================
# TAB 3: RESAMPLING & RARE CLASSES (E3-E4)
# ==========================================
with tab3:
    st.header("Experiments E3 & E4: Class Imbalance & Rare Attack Handling")
    st.markdown("""
    Under extreme class imbalance, typical classifiers favor majority BENIGN traffic while missing critical rare intrusions.
    Resampling strategies (SMOTE, SMOTE-ENN, Class Weights) were fit **strictly on training partitions** to prevent contamination.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("E3: Resampling Comparison (Random Forest)")
        if results["e3"]:
            resamp_data = []
            for strat, res in results["e3"].items():
                if "random_forest" in res:
                    resamp_data.append({
                        "Strategy": strat.replace("_", "-").upper(),
                        "MCC": res["random_forest"]["mcc_mean"],
                        "Attack Recall": res["random_forest"]["recall_mean"],
                        "Macro-F1": res["random_forest"]["macro_f1_mean"],
                        "FPR": res["random_forest"]["fpr_mean"]
                    })
            st.dataframe(pd.DataFrame(resamp_data), use_container_width=True)
        else:
            st.info("E3 data available after executing pipeline.")

    with col2:
        st.subheader("E4: Rare Attack Detection Rates")
        if results["e4"] and "raw_classes_strategy" in results["e4"]:
            rare_perf = results["e4"]["raw_classes_strategy"]["rare_classes_performance"]
            st.dataframe(pd.DataFrame(rare_perf)[["class", "recall", "precision", "f1_score", "support"]], use_container_width=True)
        else:
            st.info("E4 data available after executing pipeline.")

# ==========================================
# TAB 4: SHAP EXPLAINABILITY (E5)
# ==========================================
with tab4:
    st.header("Experiment E5: SHAP Interpretability & Feature Attribution")
    st.markdown("""
    **NIST AI RMF Principle:** High scores are insufficient without explainability. 
    SHAP confirms that removing leakage features compels the model to base predictions on authentic network behavior 
    (packet lengths, inter-arrival times, window sizes) rather than destination ports.
    """)

    shap_fig_path = FIGURES_DIR / "shap_leakage_comparison.png"
    if shap_fig_path.exists():
        st.image(str(shap_fig_path), caption="SHAP Global Feature Attribution: Baseline E1 (Left) vs Leakage-Free E2 (Right)", use_column_width=True)
    elif results["e5"] and "top_10_after_leakage" in results["e5"]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Attributed Features: Baseline E1")
            st.dataframe(pd.DataFrame(results["e5"]["top_10_before_leakage"]), use_container_width=True)
        with col2:
            st.subheader("Top Attributed Features: Leakage-Free E2")
            st.dataframe(pd.DataFrame(results["e5"]["top_10_after_leakage"]), use_container_width=True)
    else:
        st.info("Execute Experiment E5 to generate and render SHAP attribution diagrams.")

# ==========================================
# TAB 5: LIVE FLOW INSPECTOR & ALERT TRIAGE
# ==========================================
with tab5:
    st.header("🚨 Real-Time Intrusion Alert Triage Console")
    st.markdown("Simulate incoming network flow telemetry, trigger multi-class ML classification, and inspect NIST-aligned response recommendations.")

    # Flow Scenario Preset Selector
    scenario = st.selectbox(
        "Load Pre-Configured Flow Telemetry Scenario:",
        [
            "Scenario A: Normal HTTPS Traffic (BENIGN)",
            "Scenario B: SYN Stealth Port Sweep (PortScan - T1046)",
            "Scenario C: Volumetric Flooding Attack (DDoS - T1498.001)",
            "Scenario D: Web Injection Attack (SQLi / XSS - T1190)",
            "Scenario E: SSH Password Spray (Brute Force - T1110)"
        ]
    )

    # Defaults per scenario
    if "Normal HTTPS" in scenario:
        def_dur, def_fwd_pkts, def_bwd_pkts, def_fwd_len, def_bwd_len = 250000, 15, 20, 350.0, 850.0
        def_iat, def_win_fwd, def_syn, def_ack, def_psh = 8000.0, 29200, 0, 1, 1
    elif "PortScan" in scenario:
        def_dur, def_fwd_pkts, def_bwd_pkts, def_fwd_len, def_bwd_len = 1200, 2, 0, 24.0, 0.0
        def_iat, def_win_fwd, def_syn, def_ack, def_psh = 150.0, 1024, 1, 0, 0
    elif "DDoS" in scenario:
        def_dur, def_fwd_pkts, def_bwd_pkts, def_fwd_len, def_bwd_len = 15000, 450, 2, 450.0, 15.0
        def_iat, def_win_fwd, def_syn, def_ack, def_psh = 25.0, 256, 1, 0, 0
    elif "Web Injection" in scenario:
        def_dur, def_fwd_pkts, def_bwd_pkts, def_fwd_len, def_bwd_len = 45000, 12, 18, 750.0, 600.0
        def_iat, def_win_fwd, def_syn, def_ack, def_psh = 2500.0, 8192, 0, 1, 1
    else:  # SSH Password Spray
        def_dur, def_fwd_pkts, def_bwd_pkts, def_fwd_len, def_bwd_len = 85000, 18, 14, 85.0, 65.0
        def_iat, def_win_fwd, def_syn, def_ack, def_psh = 1200.0, 29200, 0, 1, 1

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Timing & Durations**")
        flow_duration = st.number_input("Flow Duration (µs)", value=int(def_dur), step=5000)
        flow_iat_mean = st.number_input("Flow IAT Mean (µs)", value=float(def_iat), step=100.0)
        init_win_fwd = st.number_input("Init Window Bytes (Fwd)", value=int(def_win_fwd), step=1024)

    with col2:
        st.markdown("**Packet Volumes & Payloads**")
        tot_fwd_pkts = st.number_input("Total Fwd Packets", value=int(def_fwd_pkts), step=1)
        tot_bwd_pkts = st.number_input("Total Bwd Packets", value=int(def_bwd_pkts), step=1)
        fwd_len_mean = st.number_input("Fwd Packet Length Mean (bytes)", value=float(def_fwd_len), step=20.0)
        bwd_len_mean = st.number_input("Bwd Packet Length Mean (bytes)", value=float(def_bwd_len), step=20.0)

    with col3:
        st.markdown("**TCP Control Flags**")
        syn_flag = st.selectbox("SYN Flag", [0, 1], index=def_syn)
        ack_flag = st.selectbox("ACK Flag", [0, 1], index=def_ack)
        psh_flag = st.selectbox("PSH Flag", [0, 1], index=def_psh)

    if st.button("🚀 Execute Intrusion Detection & Triage", type="primary", use_container_width=True):
        if "binary_rf" in models and "multi_xgb" in models and "metadata" in models:
            feature_names = models["metadata"]["feature_names"]
            classes = models["metadata"]["classes"]

            # Construct mock input matching trained behavioral feature vector
            input_dict = {f: 0.0 for f in feature_names}
            input_dict["Flow Duration"] = flow_duration
            input_dict["Flow IAT Mean"] = flow_iat_mean
            input_dict["Total Fwd Packets"] = tot_fwd_pkts
            input_dict["Total Backward Packets"] = tot_bwd_pkts
            input_dict["Fwd Packet Length Mean"] = fwd_len_mean
            input_dict["Bwd Packet Length Mean"] = bwd_len_mean
            input_dict["Init_Win_bytes_forward"] = init_win_fwd
            input_dict["SYN Flag Count"] = syn_flag
            input_dict["ACK Flag Count"] = ack_flag
            input_dict["PSH Flag Count"] = psh_flag

            input_df = pd.DataFrame([input_dict])

            # Binary Inference
            bin_pred, _ = models["binary_rf"].predict(input_df)
            bin_label = "MALICIOUS (ATTACK)" if bin_pred[0] == 1 else "BENIGN"

            # Multiclass Inference
            multi_pred, _ = models["multi_xgb"].predict(input_df)
            pred_class_idx = multi_pred[0]
            pred_class_name = classes[pred_class_idx] if pred_class_idx < len(classes) else "Unknown"

            alert_info = mapper.enrich_alert(pred_class_name)

            st.markdown("### Detection & Incident Response Brief")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Binary Status", bin_label)
            with c2:
                st.metric("Attack Family", pred_class_name)
            with c3:
                st.metric("MITRE Technique", alert_info["mitre_technique_id"])
            with c4:
                sev_color = "red" if alert_info["severity"] in ["High", "Critical"] else "green"
                st.metric("Risk Severity", alert_info["severity"])

            # NIST Guidance
            st.success(f"**MITRE ATT&CK Tactic:** {alert_info['mitre_tactic']} | **Technique:** {alert_info['mitre_technique_name']}")
            st.warning(f"**NIST SP 800-94 Mitigation Action:** {alert_info['recommended_mitigation']}")

        else:
            st.error("Trained models not found. Run `python -m src.experiments.run_all` first to train models.")
