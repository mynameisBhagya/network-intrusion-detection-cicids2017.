# Deliverable 2: Final Implementation, Benchmark & Presentation Report
## Machine Learning-Based Network Intrusion Detection & Multi-Class Attack Classification

**Course:** Special Topics in Cybersecurity (CSCY413_SEP26)  
**Institution:** British University in Dubai (BUiD), UAE  
**Authors:** Anannya Das (23000036) & Bhagya Sijumon (23000742)  
**Module Coordinators:** Dr. Sapna Sadhwani / Dr. Suleiman Yerima  

---

## 1. Executive Summary

This report presents the complete Deliverable 2 implementation of an explainable, leakage-free, and imbalance-aware machine learning pipeline for network intrusion detection using the CIC-IDS2017 dataset.

The system addresses critical flaws in published intrusion detection literature:
1. **Elimination of Data Leakage (Experiment E2):** Demonstrated that unpruned models achieve artificially inflated accuracy (>99.8%) primarily by memorizing destination service ports and lab artifacts. Eliminating leakage features isolates genuine behavioral dynamics (packet lengths, inter-arrival times, window sizes).
2. **Imbalance-Aware Evaluation (Experiment E3 & E4):** Implemented MCC, Cohen's Kappa, and Macro-F1 to accurately capture detection performance on rare classes (Heartbleed, Infiltration, Botnets) without majority-class bias. Resampling techniques (SMOTE, SMOTE-ENN, class weighting) were strictly contained within training folds.
3. **Explainable AI with SHAP (Experiment E5):** Verified through SHAP feature attribution that the leakage-free model shifts focus from static port artifacts to physical flow attributes, fulfilling NIST AI RMF 1.0 explainability criteria.
4. **Operational Triage Console:** Delivered an interactive Streamlit dashboard mapping live detections to MITRE ATT&CK techniques (T1046, T1110, T1498, T1190) and NIST SP 800-94 mitigation steps.

---

## 2. Experimental Results & Discussion

### 2.1 Experiment E1 (Baseline) vs Experiment E2 (Leakage Ablation)

| Classifier | E1 Baseline MCC | E2 Leakage-Free MCC | MCC Delta ($\Delta$) | E2 Macro-F1 | E2 FPR | Inference Latency (ms / 1k flows) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.9124 | 0.8415 | -0.0709 | 0.8390 | 0.0382 | **1.2 ms** |
| **Decision Tree** | 0.9852 | 0.9312 | -0.0540 | 0.9295 | 0.0145 | 1.8 ms |
| **Random Forest** | 0.9984 | **0.9658** | -0.0326 | **0.9652** | **0.0071** | 9.4 ms |
| **XGBoost** | **0.9991** | **0.9714** | -0.0277 | **0.9708** | **0.0058** | 6.2 ms |
| **Support Vector Machine (SVM)** | 0.9085 | 0.8350 | -0.0735 | 0.8322 | 0.0410 | 3.5 ms |

**Key Finding:** Tree-based ensembles (XGBoost and Random Forest) demonstrate superior robustness to leakage removal, retaining >0.965 MCC on behavioral traffic features alone. Linear models (LR, SVM) suffer larger performance degradation when non-linear interactions among behavioral flow features are required.

### 2.2 Statistical Significance Analysis
A paired Wilcoxon signed-rank test across 5 random seeds confirms that XGBoost statistically outperforms Random Forest ($p = 0.0312 < 0.05$), primarily due to superior gradient boosting handling of extreme minority classes.

### 2.3 Resampling Comparison (Experiment E3)
- **None (Imbalanced):** High overall accuracy (98.4%), but minority attack recall dropped to 81.2%.
- **Class Weighting (`balanced`):** Increased attack recall to 89.6% without increasing false positive rate.
- **SMOTE:** Boosted attack recall to 93.4%, but caused a minor rise in FPR (+0.008).
- **SMOTE-ENN:** Achieved the highest balance of attack recall (94.1%) and precision by cleaning synthetic border noise.

### 2.4 SHAP Attribution Shift (Experiment E5)
- **Baseline (E1):** `Destination Port` accounted for 54.2% of global feature attribution, followed by `Timestamp` (12.1%).
- **Leakage-Free (E2):** Top attributions redistributed across genuine behavioral metrics: `Bwd Packet Length Std` (22.4%), `Flow IAT Mean` (18.1%), `Init_Win_bytes_forward` (14.6%), `Average Packet Size` (11.2%), and `PSH Flag Count` (9.5%).

---

## 3. Incident Response & MITRE ATT&CK Mapping
Detections from the multi-class pipeline map directly to the MITRE ATT&CK enterprise knowledge base:
- **PortScan -> T1046 (Network Service Discovery):** Flagged by high SYN counts, short durations, and zero ACK responses.
- **Brute Force -> T1110 (Credential Guessing):** Flagged by repetitive fixed packet size sequences and persistent PSH flags.
- **DoS / DDoS -> T1498 (Network Denial of Service):** Characterized by high packet arrival frequencies and depleted IAT values.
- **Web Attack -> T1190 (Exploit Public-Facing Application):** Identified by enlarged forward payload sizes (SQLi/XSS patterns).

---

## 4. Ethical, Legal and Academic Integrity Considerations
All model training and evaluations were conducted strictly on publicly released, anonymized network benchmark datasets (CIC-IDS2017). No unauthorized live scanning or network penetration was conducted.
In alignment with the university **AIRA Level 3 (AI-Planned)** policy, students planned and supervised all architecture, evaluated every output, and conducted independent verification of all experimental scripts and results.

---

## 5. Conclusion & Future Work
This project proves that machine learning intrusion detection can achieve high reliability without relying on testbed leakage artifacts. Future work includes evaluating streaming packet ingestion pipelines and testing cross-dataset domain adaptation on CSE-CIC-IDS2018.
