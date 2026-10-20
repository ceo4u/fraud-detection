# 🛡️ Health Insurance Fraud Detector (Provider-Level CMS Claims)

An end-to-end machine learning system designed to detect healthcare provider fraud in Medicare claims data. Built with rigorous leakage prevention, stratified cross-validation, calibrated decision thresholds, and SHAP behavioral explainability.

---

## 1. Problem
Healthcare insurance fraud costs Medicare and private payers tens of billions of dollars annually. Unlike claim-level anomaly detection, the regulatory and investigative ground truth in CMS datasets exists at the **healthcare provider level**, where only **~9.35%** (506 out of 5,410) of providers are fraudulent. This extreme class imbalance renders standard accuracy misleading—predicting all providers as non-fraudulent would achieve >90% accuracy while failing to intercept a single dollar in fraudulent disbursements. This project addresses the challenge by framing provider-level risk scoring around **PR-AUC (Precision-Recall Area Under Curve)** and tuning operational decision thresholds to maximize **fraud recall** while maintaining actionable precision.

---

## 2. Dataset
Sourced from the Kaggle **Healthcare Provider Fraud Detection Analysis** benchmark, the training dataset comprises 558,211 inpatient and outpatient claims associated with 138,556 unique Medicare beneficiaries across 5,410 labeled providers:
- `data/Train_Beneficiarydata.csv`: Patient demographics (`DOB`, `DOD`, `Gender`, `Race`, `State`, `County`), Medicare Part A/B coverage months, 11 chronic condition indicators (`ChronicCond_*`), and annual reimbursement/deductible summaries.
- `data/Train_Inpatientdata.csv`: Hospital admission records (`AdmissionDt`, `DischargeDt`, `DiagnosisGroupCode`, `InscClaimAmtReimbursed`, `DeductibleAmtPaid`, `AttendingPhysician`, `OperatingPhysician`, `OtherPhysician`, ICD-9 diagnosis codes 1–10, procedure codes 1–6).
- `data/Train_Outpatientdata.csv`: Clinic and outpatient claim records with diagnosis/procedure codes and reimbursement amounts.
- `data/Train.csv`: Provider-level ground truth labels (`Provider`, `PotentialFraud` with values `'Yes'` or `'No'`).

*Note: Unlabeled test splits (`Test_*.csv`) are excluded to avoid ungrounded evaluation.*

---

## 3. Approach
1. **Unified Claims Ingestion (`src/data_prep.py`)**: Concatenates inpatient and outpatient claims, joins patient demographics by `BeneID`, parses calendar dates, and computes claim-level metrics (patient age at service, claim duration, hospital stay length, chronic disease burden, renal disease, beneficiary mortality, and missing physician flags).
2. **Behavioral Feature Engineering (`src/features.py`)**: Aggregates claim patterns to provider level (5,410 rows, 30 features). Computes volume metrics, reimbursement distributions, clinical diversity scores (unique diagnosis/procedure code ratios), beneficiary concentration (`reimb_per_bene`, `claims_per_bene`), and physician operational signatures (`missing_operating_ratio`, `physician_per_bene_ratio`).
3. **Dual Model Architecture (`src/train.py`)**: 
   - **Baseline**: Scaled Logistic Regression with balanced class weighting.
   - **Primary Model**: Gradient-boosted trees (`XGBClassifier`) with `scale_pos_weight` and PR-AUC optimization.
4. **Leakage-Free OOF Threshold Tuning**: 5-fold Stratified K-Fold cross-validation generates out-of-fold (OOF) probability estimates. Three candidate thresholds are derived strictly from OOF:
   - *Recall-First Threshold*: Lowest boundary where Precision $\ge$ 0.30.
   - *Balanced Threshold (Production)*: Lowest boundary where Precision $\ge$ 0.50 (1 in 2 alerts is true fraud; defensible false-alarm rate).
   - *F1-Optimal Threshold*: Maximizes the harmonic mean of precision and recall.
5. **Explainability Engine (`src/explain.py`)**: Leverages `shap.TreeExplainer` to extract provider-level waterfall plots and global feature attribution vectors.
6. **Auditor Dashboard (`app.py`)**: Interactive Streamlit application with live threshold slider, demo provider shortcut, CSV upload capability, individual waterfall explanations, and performance curves.

---

## 4. Results Table (Held-Out Test Set: 1,082 Providers)

Evaluated on a strictly isolated 20% stratified test split (101 Fraudulent, 981 Non-Fraudulent providers):

### Threshold Strategies on Held-Out Test Set (XGBoost Classifier — PR-AUC: 0.7840, ROC-AUC: 0.9695)

| Threshold Strategy | Threshold | Precision | Recall | F1 Score | Rationale & Trade-off |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Recall-first (precision $\ge$ 0.30)** | 0.0331 | 0.2841 | 0.9901 | 0.4415 | Catches 99% of fraud, but 72% false alarm rate. |
| **Balanced (precision $\ge$ 0.50) ← PRODUCTION** | **0.2780** | **0.4972** | **0.8911** | **0.6383** | **Defensible 1-in-2 audit ratio while catching ~89% of fraud.** |
| **F1-optimal** | 0.5928 | 0.6638 | 0.7624 | 0.7097 | Conservative auditing; misses ~24% of fraud rings. |

> **Design Rationale on Metrics:** Accuracy is deliberately not reported — with 9.35% fraud prevalence, a trivial "predict all non-fraud" classifier would achieve **90.65% accuracy** while capturing zero fraud dollars. We optimize **PR-AUC** and precision-constrained recall instead.

### Top 10 SHAP Feature Drivers:
1. `total_reimb` (1.7488) — Total dollar reimbursements processed.
2. `max_claim_duration` (0.8690) — Abnormally elongated claim duration periods.
3. `claims_per_bene` (0.4232) — Unusually high claim repetition per beneficiary.
4. `missing_operating_ratio` (0.2767) — Missing operating physician on inpatient claims.
5. `avg_deductible` (0.2507) — Elevated average deductible charges.
6. `max_diag_per_claim` (0.2409) — Saturation of maximum diagnosis code slots.
7. `reimb_per_bene` (0.2343) — Disproportionate reimbursement extracted per patient.
8. `std_reimb` (0.1991) — High variance and spike volatility in claim amounts.
9. `avg_chronic` (0.1920) — Patient cohort chronic condition concentration.
10. `dead_bene_ratio` (0.1849) — Claims filed involving deceased beneficiaries.

---

## 5. Demo Walkthrough (90-Second Script)

Follow these 6 steps to present the end-to-end system during a demo:

1. **Sidebar Setup**: Keep "Data Source" on *Sample data* and observe the slider defaulted to the production threshold (`0.28`).
2. **Explain the Imbalance**: Point to the KPI cards in Tab 1: *"Out of 5,410 providers, fraud is only ~9.35%. A generic model would predict non-fraud for all and score 91% accuracy. We tune for recall at an actionable 50% precision."*
3. **Ranked Provider Scans**: Scroll through the Tab 1 table. Point out the top-ranked providers color-coded **High Risk** with probabilities $>95\%$.
4. **Instant Case Inspection**: Click **Tab 2 (Explain provider)** and press the **⭐ Load Demo Provider** button. This automatically loads validated fraud case `PRV52340` (Fraud Probability: `99.87%`).
5. **SHAP Factor Breakdown**: Show the interactive Waterfall chart. Point to the top driving signals:
   - High total reimbursement volume (`+$2.54M`, +2.99 SHAP)
   - Disproportionate claim count (`1,743` claims, +1.38 SHAP)
   - Max claim duration outliers (`35` days, +1.15 SHAP)
6. **Audit Metrics & False Negative Alert**: Switch to **Tab 3 (Model Performance)**. Highlight the Confusion Matrix where False Negatives (FN) are outlined in red warning boxes: *"At our balanced threshold, the model catches 90 out of 101 fraud cases on unseen test data."*

---

## 6. How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Complete Pipeline Execution
```bash
# 1. Ingest raw CSVs, compute derived columns, export data/claims_processed.parquet
python src/data_prep.py

# 2. Aggregate to provider level and export models/features.parquet
python src/features.py

# 3. Train models, run 5-fold CV, tune thresholds, evaluate on test set
python src/train.py

# 4. Find curated demo provider case
python src/find_demo_provider.py

# 5. Generate SHAP global summary, top 10 bar plots, and precomputed vectors
python src/explain.py

# 6. Launch interactive auditor dashboard (Streamlit)
python -m streamlit run app.py

# 7. (Optional) Run Enterprise Java Spring Boot + FastAPI Architecture
# Backend (FastAPI on :8000):
powershell -ExecutionPolicy Bypass -File run_backend.ps1
# Frontend (Spring Boot 3.2 on :8081):
powershell -ExecutionPolicy Bypass -File run_frontend.ps1
```

---

## 7. Dual Frontend Architecture
This repository provides two deployment targets:
1. **Interactive Streamlit App (`app.py`)**: Accessible on `http://localhost:8501`. Features dynamic threshold adjustments, SHAP waterfall attributions, benchmark confusion matrices, and the **📤 Batch Score** raw claims pipeline.
2. **Enterprise Spring Boot + FastAPI (`fraudguard-java/`)**: Microservice architecture connecting a Java 17 Spring Boot presentation layer (`http://localhost:8081`) via HTTP REST proxy to a high-throughput Python FastAPI inference server (`http://localhost:8000`).

---

## 8. Key Design Decisions
- **Provider-Level Aggregation**: Individual claims lack fraud labels; aggregating behavioral patterns (diversity, per-beneficiary rates, network concentration) reflects the true operational unit of investigation.
- **No SMOTE / Synthetic Oversampling**: SMOTE generates artificial observations that distort the joint covariance of heavily skewed Medicare billing distributions. Instead, we use native tree weighting (`scale_pos_weight`) and empirical OOF threshold tuning.
- **PR-AUC as Primary Metric**: ROC-AUC can be deceptively optimistic under extreme class imbalance due to large true negative counts. PR-AUC directly tracks precision across high-recall operating regimes.
- **OOF Threshold Tuning (No Test Leakage)**: Thresholds are computed exclusively from out-of-fold cross-validation predictions on `X_train`, preserving strict independence of the held-out test split.
- **Calibrated Balanced Threshold**: Selecting `0.2780` ensures 1 in 2 flagged providers is true fraud, maximizing compliance team efficiency while intercepting 89.11% of fraudulent providers.

---

## 8. Limitations
- **Single Benchmark Dataset**: Models are trained on historical CMS Medicare data and require recalibration for commercial private insurers.
- **Provider-Level Labels**: Labeling an entire provider as fraudulent does not distinguish between individual bad-actor physicians versus systemic organizational billing fraud.
- **Correlation vs. Causation**: High reimbursement volume naturally correlates with large regional healthcare facilities; auditing rules must account for hospital bed capacity.

---

## 9. Future Work
- **Graph Neural Networks (GNNs)**: Constructing bipartite graphs linking Beneficiary—Provider—Physician to detect collusive referral rings.
- **Temporal Validation**: Evaluating models on time-based train/test splits (e.g., train on Q1–Q3, evaluate on Q4) to capture drift in fraudulent billing tactics.
- **Real-Time Streaming Scoring**: Serving lightweight inferences via a REST API integrated directly into claims adjudication gateways before payments are cleared.
