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

---

## 3. Technology Stack & Directory Structure

```text
CARDIO-X/
├── .venv/                      # Isolated virtual environment
├── app/                        # Streamlit application
│   ├── streamlit_app.py        # Application entrypoint & navigation
│   ├── components/             # Reusable UI banners, headers, and footers
│   ├── pages/                  # Application pages (Home, NHANES, Framingham, etc.)
│   │   └── nhanes_screening.py # Interactive NHANES non-laboratory screening module
│   └── utils/                  # Application utilities
│       ├── model_loader.py     # Safe, cached model and metadata loader
│       └── screening_engine.py # Validation engine and inference execution
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
├── tests/                      # Automated test suite (76 unit/integration tests)
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

1. **Demographics & Socioeconomics:**
   - `age_years`: Age (18–80 years).
   - `sex`: Biological sex (Male, Female).
   - `race_ethnicity`: Self-reported race/ethnicity category.
   - `education_level`: Highest attained educational attainment.
   - `family_income_poverty_ratio`: Family income-to-poverty ratio (0.00–5.00).

2. **Anthropometrics & Body Composition:**
   - `standing_height_cm`: Standing height (100–220 cm).
   - `weight_kg`: Measured body weight (30–250 kg).
   - `bmi_kg_m2`: Automatically computed Body Mass Index ($kg/m^2$).
   - `waist_circumference_cm`: Waist circumference (50–180 cm).
   - `waist_to_height_ratio`: Automatically derived central adiposity index ($cm/cm$).
   - `weight_change_1yr_kg`: Self-reported weight change over the preceding 12 months (-50 to +50 kg).

3. **Hemodynamics & Vital Signs:**
   - `avg_systolic_bp_mmhg`: Average resting systolic blood pressure (70–240 mmHg).
   - `avg_diastolic_bp_mmhg`: Average resting diastolic blood pressure (40–140 mmHg).
   - `pulse_pressure_mmhg`: Automatically derived hemodynamic marker ($\text{SBP} - \text{DBP}$).
   - `avg_pulse_bpm`: Resting heart rate (40–160 bpm).
   - `bp_medication`: Currently taking prescribed antihypertensive medication (Yes/No).

4. **Lifestyle & Behavioral Risk Factors:**
   - `smoking_status`: Smoking history (Never, Former, Current).
   - `total_ltp_met_minutes_week`: Leisure-time physical activity in MET-minutes/week.
   - `sedentary_minutes_day`: Self-reported sedentary duration in minutes/day.
   - `alcohol_drinks_per_week`: Standard alcoholic beverages consumed per week.
   - `sleep_hours_weekday`: Average weekday sleep duration (2–16 hours).

5. **Medical History & General Health:**
   - `diabetes_diagnosed`: Prior formal diagnosis of diabetes (Yes/No).
   - `general_health`: Self-assessed overall health status (Excellent, Very Good, Good, Fair, Poor).
   - `phq9_score`: Patient Health Questionnaire-9 depression screening score (0–27).
   - `coronary_heart_disease`: Prior history of coronary heart disease (Yes/No).
   - `stroke`: Prior history of cerebrovascular accident / stroke (Yes/No).

### 5.4 Understanding the Screening Output
When you click **"Run Non-Laboratory Screening"**, the system evaluates the 24-feature vector through the pre-trained `models/nhanes_screening_core.joblib` pipeline:

- **Screening Score:** A calibrated probability output between 0.0% and 100.0% estimating phenotypic resemblance to individuals with hypercholesterolemia in the CDC NHANES reference population.
- **Operating Threshold ($\tau^* = 0.46$):** Derived strictly via 5-fold cross-validation out-of-fold training predictions to achieve a screening sensitivity $\ge 80\%$. On the held-out test cohort, this cutoff delivers **82.2% sensitivity** and **61.5% specificity** (ROC-AUC: 0.7932).
- **Classification Banners:**
  - **Higher Screening Signal (Score $\ge 0.46$):** The patient profile displays biometric and metabolic patterns frequently co-occurring with elevated cholesterol or lipid management in the general population.
  - **Lower Screening Signal (Score $< 0.46$):** The profile presents lower statistical resemblance to phenotypic hypercholesterolemia.

### 5.5 Why Confirmatory Laboratory Lipid Testing Is Essential
Non-laboratory risk screening serves exclusively as a **prioritization and triage mechanism**:
- **False Positives:** Individuals may have elevated scores due to age, obesity, or hypertension without having high serum cholesterol.
- **False Negatives:** Individuals with genetic dyslipidemias (e.g., Familial Hypercholesterolemia) often appear lean, active, and normotensive, yet have critically high circulating LDL-C.
- **Definitive Diagnosis:** A definitive lipid assessment requires a standardized venous blood draw measuring Total Cholesterol, LDL-C, HDL-C, and Triglycerides.

---

## 6. Verification and Reproducibility
All 76 automated regression tests can be run at any time to verify system integrity:
```bash
python -m pytest tests/ -v
```
All model weights, parameters, and operating cutoffs remain immutable and are loaded directly from the validated artifacts in `models/`.
