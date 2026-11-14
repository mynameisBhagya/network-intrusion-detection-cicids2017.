# Machine Learning-Based Network Intrusion Detection and Multi-Class Attack Classification
## Deliverable 1: Introduction, Literature Review and Methodology Report

**Course:** Special Topics in Cybersecurity (CSCY413_SEP26)  
**Institution:** British University in Dubai (BUiD), UAE  
**Authors:** Anannya Das (23000036) & Bhagya Sijumon (23000742)  

---

### Abstract
The most dependable models for machine learning-based network intrusion detection using CIC-IDS2017 are tree-based ensemble classifiers such as Random Forest and XGBoost; however, their real-world effectiveness largely depends on how class imbalance and data leakage are handled. This report provides an empirical comparison across five standard algorithms: Logistic Regression, Decision Tree, Random Forest, XGBoost, and Support Vector Machine (SVM), in both binary and multi-class classification settings. Major issues of CIC-IDS2017 are addressed: class imbalance, attack rarity, feature leakage, and minority noise. The results indicate that MCC and Cohen's Kappa provide significantly better fidelity than accuracy under imbalance, and that SHAP-based model explanations must only be relied upon after non-behavioral leakage features are purged.

---

### 1. Introduction & Problem Statement
Intrusion Detection Systems (IDS) serve as essential controls in modern enterprise defense (NIST SP 800-94). While CIC-IDS2017 is one of the most widely cited benchmark datasets, published literature frequently suffers from four critical flaws:
1. **Multiclass Class Imbalance:** Severe dominance of benign traffic (>80%) renders overall accuracy misleading.
2. **Data Leakage Artifacts:** Inclusion of features such as `Destination Port`, `Timestamp`, and IP identifiers enables models to memorize specific testbed configurations rather than generalizable attack signatures.
3. **Metric Sensitivity:** Reliance on raw accuracy masks false alarm rates and fails to capture rare attack performance.
4. **Superficial Interpretability:** Feature importance analyses are rarely examined before and after leakage removal.

Our project addresses these gaps through an end-to-end, leakage-resistant, imbalance-aware, and explainable machine learning pipeline.

---

### 2. Literature Review & Identified Gaps

| Identified Gap | Literature Evidence | Project Response |
|---|---|---|
| **Leakage rarely controlled** | Gharghi & Dehghani (2026); Arcos-Argudo et al. (2025) note testbed artifact memorization. | Purges port/IP/timestamp identifiers; reports baseline vs leakage-free delta. |
| **Inconsistent imbalance remedies** | Bacevicius & Paulauskaite-Taraseviciene (2023); Singh et al. (2025). | Evaluates SMOTE, SMOTE-ENN, and class weights strictly on training splits. |
| **Misleading metric selection** | Luque et al. (2019) prove accuracy distortion under imbalance. | Prioritizes MCC, Cohen's Kappa, and Macro-F1 alongside operational latency. |
| **SHAP unverified against leakage** | Ahmed et al. (2024); Zhang et al. (2026). | Compares SHAP attributions before and after leakage removal. |
| **Non-comparable study rankings** | Chen & Lyu (2020); Tama & Rhee (2017); Soon et al. (2024). | Employs a unified experimental protocol with fixed seeds. |

---

### 3. Frameworks & Standards Alignment
- **NIST SP 800-94:** Guide to Intrusion Detection and Prevention Systems (management of false alarms and detection operationalization).
- **NIST AI RMF 1.0:** Trustworthy AI requirements: validity, reliability, and explainability.
- **MITRE ATT&CK:** Enterprise matrix mapping:
  - PortScan -> T1046 (Network Service Discovery)
  - Brute Force -> T1110 (Brute Force: Credential Guessing)
  - DoS/DDoS -> T1498 (Network Denial of Service)
  - Web Attack -> T1190 (Exploit Public-Facing Application)
  - Botnet -> T1071 (Application Layer Protocol: C2)
  - Infiltration -> T1210 (Exploitation of Remote Services)
  - Heartbleed -> T1212 (Exploitation for Credential Access)

---

### 4. Experimental Plan (E1 – E5)
- **E1 (Baseline):** 5 classifiers evaluated on unpruned feature space across 5 random seeds.
- **E2 (Leakage Ablation):** 5 classifiers evaluated on leakage-free behavioral features; performance delta quantified.
- **E3 (Resampling):** None vs Class Weighting vs SMOTE vs SMOTE-ENN fit strictly on training partitions.
- **E4 (Rare Classes):** Study detection rates on rare attacks (Heartbleed, Infiltration, Web Attacks).
- **E5 (Explainability):** Tree-based SHAP analysis before vs after leakage removal.
