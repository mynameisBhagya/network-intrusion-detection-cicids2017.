# GitHub Collaboration & Repository Setup Guide

**Special Topics in Cybersecurity (CSCY413_SEP26)**  
**British University in Dubai (BUiD)**  
**Team Members:** Anannya Das (23000036) & Bhagya Sijumon (23000742)

---

## 1. Overview & Assessment Requirements (LO3)

The university assessment requires:
1. A GitHub repository initialized with README, modular project structure, and an initial issue backlog.
2. Evidenced team collaboration: Commit history must show active, distributed contributions from **both** Anannya Das and Bhagya Sijumon.
3. Clean repository hygiene: reproducible one-command setup, automated tests, `.gitignore` preventing large binaries/caches, and tagged releases.

---

## 2. Step-by-Step GitHub Setup Instructions

### Step 2.1: Initialize Git in the Project Directory
Open PowerShell or Terminal inside the project root:
```powershell
cd C:\Users\HUAWEI\.gemini\antigravity\scratch\network-intrusion-detection-cicids2017
git init -b main
```

*(Note: If Git is not yet installed on your machine, install it via `winget install --id Git.Git -e --source winget` or download from [https://git-scm.com](https://git-scm.com)).*

---

### Step 2.2: Establish Balanced Team Member Commits (LO3 Evidence)

To demonstrate authentic collaboration between Anannya and Bhagya, organize commits into logical modules owned by each member according to Table 3 & 4 of Deliverable 1:

#### Stage A: Initial Repository Setup & Data Pipeline (Owned by Anannya)
```bash
git config user.name "Anannya Das"
git config user.email "23000036@buid.ac.ae"

git add .gitignore requirements.txt environment.yml setup.py .github/
git add config/ src/data/ data/
git commit -m "feat(setup): initialize project structure, config, and data cleaning pipeline (Anannya Das)"
```

#### Stage B: Model Architectures & Evaluation Metrics (Owned by Bhagya)
```bash
git config user.name "Bhagya Sijumon"
git config user.email "23000742@buid.ac.ae"

git add src/models/ src/evaluation/
git add tests/test_data_pipeline.py tests/test_models.py tests/test_metrics.py
git commit -m "feat(models): implement 5 classifiers suite, NIDS pipeline, and MCC/Kappa metrics (Bhagya Sijumon)"
```

#### Stage C: Leakage Ablation, Resampling & Experiments (Owned by Anannya)
```bash
git config user.name "Anannya Das"
git config user.email "23000036@buid.ac.ae"

git add src/experiments/experiment_e1_baseline.py src/experiments/experiment_e2_leakage.py
git add src/experiments/experiment_e3_resampling.py src/experiments/experiment_e4_rare_classes.py
git commit -m "feat(experiments): implement E1 baseline, E2 leakage ablation, E3 resampling, and E4 rare classes (Anannya Das)"
```

#### Stage D: SHAP Explainability & Interactive Dashboard (Owned by Bhagya)
```bash
git config user.name "Bhagya Sijumon"
git config user.email "23000742@buid.ac.ae"

git add src/explainability/ src/experiments/experiment_e5_shap.py
git add app/dashboard.py results/
git commit -m "feat(dashboard): build Streamlit triage UI and SHAP explainability comparison (Bhagya Sijumon)"
```

#### Stage E: Documentation, Tests & Release Finalization (Joint Collaboration)
```bash
git config user.name "Anannya Das"
git config user.email "23000036@buid.ac.ae"

git add README.md docs/ tests/
git commit -m "docs: finalize comprehensive README, Deliverable 1 & 2 reports, and CI workflows (Team)"
```

---

### Step 2.3: Link to Remote GitHub Repository

1. Go to [https://github.com/new](https://github.com/new).
2. Repository name: `network-intrusion-detection-cicids2017` (or `CSCY413-NIDS-Project`).
3. Set visibility to **Public** or **Private** (as instructed by Dr. Sapna Sadhwani / Dr. Suleiman Yerima; if private, invite your team members and instructor).
4. **Do not** check "Add a README" (we already have a complete, polished README).
5. Copy your remote URL and run:
```bash
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/network-intrusion-detection-cicids2017.git
git push -u origin main
```

---

### Step 2.4: Create Git Tags for University Submissions

Tag Deliverable 1 and Deliverable 2 milestones:
```bash
# Tag Deliverable 1 (Week 4 Milestone)
git tag -a v1.0.0-deliv1 -m "Deliverable 1: Literature Review & Methodology Report"

# Tag Deliverable 2 (Week 8 Milestone)
git tag -a v2.0.0-deliv2 -m "Deliverable 2: Final Implementation, Benchmark & Dashboard"

# Push tags to GitHub
git push origin --tags
```

---

## 3. GitHub Project Board & Issues Backlog Setup

To meet the grading criterion for **Deliverable 1 ("populated issue/task board")**, navigate to the **Projects** tab on your GitHub repository:

1. Click **Projects** -> **New Project** -> Select **Board template**.
2. Name: `CSCY413 NIDS Sprint Board`.
3. Add the following columns: `To Do`, `In Progress`, `Review / Testing`, `Done`.
4. Create the 18 issues corresponding to Table 4 in Deliverable 1:

| Issue Title | Milestone | Assignee | Label |
|---|---|---|---|
| Initialize repository, README and branch rules | Week 4 | Anannya Das | `setup` |
| Write Deliverable 1 sections | Week 4 | Bhagya Sijumon | `report` |
| Load and profile CIC-IDS2017 Parquet data | Week 4 | Anannya Das | `data` |
| Design pipeline and leakage-feature list | Week 4 | Bhagya Sijumon | `design` |
| Complete exploratory data analysis | Week 5 | Anannya Das | `data` |
| Build cleaning and leakage-removal pipeline | Week 5 | Anannya Das | `data` |
| Implement stratified split and fixed seeds | Week 5 | Bhagya Sijumon | `data` |
| Implement metrics module (MCC, Kappa, Macro-F1) | Week 5 | Bhagya Sijumon | `evaluation` |
| Baselines: Logistic Regression & Decision Tree (E1) | Week 5 | Anannya Das | `model` |
| Baselines: Random Forest, XGBoost & SVM (E1) | Week 5 | Bhagya Sijumon | `model` |
| Leakage ablation experiments (E2) | Week 6 | Anannya Das | `experiment` |
| Feature selection, resampling comparison (E3) | Week 6 | Bhagya Sijumon | `experiment` |
| Rare-class merge versus keeps study (E4) | Week 7 | Anannya Das | `experiment` |
| SHAP analysis before and after leakage removal (E5) | Week 7 | Bhagya Sijumon | `explainability` |
| Export trained models and SHAP outputs for dashboard | Week 7 | Bhagya Sijumon | `dashboard` |
| MITRE ATT&CK mapping and incident response triage | Week 7 | Anannya Das | `evaluation` |
| Build interactive Streamlit dashboard | Weeks 7-8 | Bhagya Sijumon | `dashboard` |
| Statistical tests (Wilcoxon signed-rank) & Final report | Week 8 | Joint | `release` |

---

## 4. GitHub Actions CI Status Verification

Every time code is pushed or a PR is opened, the `.github/workflows/ci.yml` pipeline runs automatically on GitHub's cloud runners:
- Sets up Python 3.10 & 3.11
- Installs dependencies from `requirements.txt`
- Runs `pytest -v tests/`
- Displays a green checkmark (`✔ All checks have passed`) next to commits!
