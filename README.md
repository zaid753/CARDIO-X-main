# CARDIO-X: An Intelligent Cardiovascular Risk Screening and Lipid Profile Analysis System

## 1. Project Overview
CARDIO-X is an academic healthcare informatics and machine learning system developed to evaluate non-invasive cardiovascular risk screening and longitudinal lipid profile analysis. The project integrates two independent epidemiological datasets:

1. **CDC NHANES (2021–2023):** Used for non-invasive, non-laboratory screening for phenotypic hypercholesterolemia (Total Cholesterol $\ge 240$ mg/dL or active cholesterol pharmacotherapy).
2. **Framingham Heart Study Cohort:** Used for longitudinal 10-year incident cardiovascular disease (CVD) risk assessment.

> [!IMPORTANT]
> **Academic Research Prototype Disclaimer:** CARDIO-X is developed strictly for educational and scientific research benchmarking. It is **not** an FDA-cleared medical device, diagnostic test, or clinical decision support system. It must not be used as a substitute for professional medical advice, clinical diagnosis, or laboratory blood testing.

---

## 2. Project Architecture & Development Status

- **Phase 1: Environment & Dataset Restoration** — Completed. Python 3.11+ / 3.13 isolated `.venv`, dataset integrity checks, and dependency management.
- **Phase 2A: NHANES Screening Preprocessing Pipeline** — Completed. Non-invasive feature selection (24 core predictors), Target Option B derivation, and leakage prevention.
- **Phase 2B: Model Training, Evaluation & Methodology Validation** — Completed. Rigorous cross-validation, held-out evaluation, calibration, threshold tuning, and model persistence under `models/`.
- **Phase 3A: Streamlit Application Foundation** — Completed. Multi-page modular UI architecture, centralized model loader, and self-test diagnostics.
- **Phase 3B: Interactive NHANES Cholesterol Screening Module** — Completed. Interactive clinical non-laboratory screening interface with real-time validation, risk tiering, and clinical next-step guidance.
- **Phase 3C: Interactive Framingham 10-Year CVD Risk Prediction Module** — Completed. Interactive primary prevention risk prediction interface with 14 clinical covariates, physiological validation, derived hemodynamics (pulse pressure, MAP), calibrated operating threshold ($\tau^* = 0.40$), and academic governance drawer.

---

## 3. Technology Stack & Directory Structure

```text
CARDIO-X/
├── .venv/                      # Isolated virtual environment
├── app/                        # Streamlit application
│   ├── streamlit_app.py        # Application entrypoint & navigation
│   ├── components/             # Reusable UI banners, headers, and footers
│   ├── pages/                  # Application pages
│   │   ├── dashboard.py        # System health and model verification hub
│   │   ├── nhanes_screening.py # Interactive NHANES non-laboratory screening module
│   │   └── framingham_risk.py  # Interactive Framingham 10-year CVD risk engine
│   └── utils/                  # Application utilities
│       ├── model_loader.py     # Safe, cached model and metadata loader
│       ├── screening_engine.py # NHANES validation and inference engine
│       └── framingham_engine.py# Framingham validation and inference engine
├── data/                       # Local datasets (excluded from version control)
├── models/                     # Serialized scikit-learn pipelines & metadata
│   ├── nhanes_screening_core.joblib
│   ├── nhanes_screening_full.joblib
│   ├── framingham_risk.joblib
│   └── model_metadata.json
├── reports/                    # Model evaluation reports and audit figures
├── src/                        # Core algorithmic and preprocessing modules
│   ├── models.py               # Model training, calibration, and evaluation
│   ├── nhanes_preprocessing.py # NHANES cohort construction and pipelines
│   ├── preprocessing.py        # Framingham pipeline and cohort definitions
│   └── data_quality.py         # Data integrity and validation functions
├── tests/                      # Automated test suite (92 unit/integration tests)
└── requirements.txt            # Python dependencies
```

---

## 4. Setup & Installation

### 4.1 Activate Virtual Environment
```bash
source .venv/bin/activate
```
*(On Windows: `.\.venv\Scripts\Activate.ps1`)*

### 4.2 Install Dependencies
```bash
pip install -r requirements.txt
```

### 4.3 Run Test Suite
To verify the entire project and application regression suite:
```bash
python -m pytest tests/ -v
```

---

## 5. Interactive NHANES Screening Module (Phase 3B)

### 5.1 Launching the Application
Launch the Streamlit web interface from the project root:
```bash
streamlit run app/streamlit_app.py
```
By default, the application runs locally at `http://localhost:8501`.

### 5.2 Navigation
Use the sidebar navigation menu under **"CARDIO-X Modules"** to select:
👉 **NHANES Screening (Phase 3B)**

### 5.3 The 24 Required Non-Laboratory Inputs
The screening module requires exactly 24 non-invasive clinical, biometric, socioeconomic, and lifestyle inputs partitioned across 5 clinical panels:
- **Demographics & Socioeconomics:** `age_years`, `sex`, `race_ethnicity`, `education_level`, `family_income_poverty_ratio`.
- **Anthropometrics & Body Composition:** `bmi_kg_m2`, `waist_circumference_cm`, `waist_to_height_ratio`, `weight_change_1yr_kg`.
- **Hemodynamics & Vital Signs:** `avg_systolic_bp_mmhg`, `avg_diastolic_bp_mmhg`, `pulse_pressure_mmhg`, `avg_pulse_bpm`, `bp_medication`.
- **Lifestyle & Behavioral Risk Factors:** `smoking_status`, `total_ltp_met_minutes_week`, `sedentary_minutes_day`, `alcohol_drinks_per_week`, `sleep_hours_weekday`.
- **Medical History & General Health:** `diabetes_diagnosed`, `general_health`, `phq9_score`, `coronary_heart_disease`, `stroke`.

### 5.4 Understanding the Screening Output
- **Screening Score:** A calibrated probability output between 0.0% and 100.0% estimating phenotypic resemblance to individuals with hypercholesterolemia in the CDC NHANES reference population.
- **Operating Threshold ($\tau^* = 0.46$):** Derived strictly via 5-fold cross-validation out-of-fold training predictions to achieve a screening sensitivity $\ge 80\%$. On the held-out test cohort, this cutoff delivers **77.90% sensitivity**, **68.58% specificity**, and **72.12% accuracy** (ROC-AUC: **0.8205**).
- **Classification Banners:**
  - **Higher Screening Signal (Score $\ge 0.46$):** The patient profile displays biometric and metabolic patterns frequently co-occurring with elevated cholesterol or lipid management in the general population.
  - **Lower Screening Signal (Score $< 0.46$):** The profile presents lower statistical resemblance to phenotypic hypercholesterolemia.

### 5.5 Why Confirmatory Laboratory Lipid Testing Is Essential
Non-laboratory risk screening serves exclusively as a **prioritization and triage mechanism**. Definite clinical lipid management requires standardized venous blood sampling for Total Cholesterol, LDL-C, HDL-C, and Triglycerides.

---

## 6. Interactive Framingham 10-Year CVD Risk Module (Phase 3C)

### 6.1 Purpose & Target
The Framingham Risk Module implements an interactive primary prevention risk estimation tool predicting first incident hard cardiovascular disease within 10 years (`TARGET_10Y_CVD`) evaluated on the primary prevention cohort ($N = 3,983$, 9.26% 10-year event rate) of the Framingham Heart Study.

### 6.2 Navigation
Use the sidebar navigation menu under **"CARDIO-X Modules"** to select:
👉 **Framingham 10-Year CVD Risk (Phase 3C)**

### 6.3 The 14 Required Baseline Inputs
The module accepts exactly 14 baseline clinical covariates:
1. `AGE`: Age in years (30–85).
2. `SEX`: Biological sex (0 = Female, 1 = Male).
3. `CURSMOKE`: Current cigarette smoker (0 = No, 1 = Yes).
4. `CIGPDAY`: Cigarettes smoked per day (0 if non-smoker, 1–100 if smoker).
5. `educ`: Educational level (1: 0–11 years, 2: High school / GED, 3: Some college, 4: College graduate).
6. `BPMEDS`: Prescribed antihypertensive medication (0 = No, 1 = Yes).
7. `PREVHYP`: Prevalent hypertension history (0 = No, 1 = Yes).
8. `HEARTRTE`: Resting heart rate in beats per minute (35–180).
9. `TOTCHOL`: Total serum cholesterol in mg/dL (100–600).
10. `BMI`: Body Mass Index in $kg/m^2$ (12.0–70.0).
11. `DIABETES`: History of diagnosed diabetes (0 = No, 1 = Yes).
12. `GLUCOSE`: Casual or fasting blood glucose in mg/dL (40–500).
13. `SYSBP`: Systolic blood pressure in mmHg (70–240).
14. `DIABP`: Diastolic blood pressure in mmHg (40–140).

### 6.4 Saved Model & Calibrated Operating Threshold
- **Pipeline Artifact:** `models/framingham_risk.joblib` (L2-regularized Logistic Regression with `RobustScaler` on continuous features).
- **Calibrated Operating Threshold ($\tau^* = 0.40$):** Derived via training-set 5-fold cross-validation out-of-fold predictions to prevent sensitivity collapse and achieve $\ge 80\%$ recall.
- **Held-Out Test Performance ($N = 797$, 74 events):**
  - Test ROC-AUC: **0.7648**
  - Test Sensitivity ($\tau^* = 0.40$): **81.08%**
  - Test Specificity ($\tau^* = 0.40$): **56.85%**
  - Test Accuracy ($\tau^* = 0.40$): **59.10%**
  - Test F1 Score ($\tau^* = 0.40$): **0.2691**

### 6.5 Derived Hemodynamics
The module automatically derives and displays key hemodynamic parameters:
- **Pulse Pressure (PP):** $\text{SYSBP} - \text{DIABP}$ (indicator of arterial stiffness).
- **Mean Arterial Pressure (MAP):** $\text{DIABP} + (\text{PP} / 3)$ (indicator of systemic perfusion pressure).

### 6.6 Prediction Interpretation & Clinical Next Steps
- **Higher Risk Signal (Score $\ge 0.40$):** The patient profile exhibits cardiovascular risk factor clustering statistically associated with higher risk of incident hard CVD within 10 years.
- **Lower Risk Signal (Score $< 0.40$):** Score remains below the calibrated sensitivity cutoff.
- **Next-Step Guidance:** Emphasizes scheduling a clinical consultation with a physician, undergoing comprehensive formal cardiovascular risk assessment (e.g., ACC/AHA PCE, coronary artery calcium scoring, advanced lipid panels), and discussing individualized primary prevention strategies.

### 6.7 Academic Methodology & Limitations
- **Landmark Classifier:** Evaluates individuals in a landmark primary prevention cohort with complete 10-year follow-up or prior event. It is **not** a competing-risk survival model.
- **Exclusion Criteria:** Excludes individuals with prior baseline cardiovascular disease, stroke, or transient ischemic attack.
- **Not an Official Calculator:** This is an academic research prototype and is **not** a clinically validated replacement for the ACC/AHA Pooled Cohort Equations or official Framingham Risk Score calculators.

---

## 7. Verification and Reproducibility
All 92 automated regression tests can be run at any time to verify system integrity:
```bash
python -m pytest tests/ -v
```
All model weights, scalers, imputers, and operating thresholds remain strictly immutable and are loaded directly from the validated artifacts in `models/`.
