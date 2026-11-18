"""
CARDIO-X: Interactive NHANES Non-Laboratory Cholesterol Screening Module.

Implements the user-facing screening form, physiological validation, model prediction,
score presentation, clinical next-step guidance, and academic prototype governance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import streamlit as st
import pandas as pd

from app.utils.screening_engine import (
    NHANES_CORE_PREDICTORS,
    VALID_SEX_CATEGORIES,
    VALID_RACE_ETHNICITY_CATEGORIES,
    VALID_EDUCATION_CATEGORIES,
    VALID_SMOKING_CATEGORIES,
    VALID_GENERAL_HEALTH_CHOICES,
    validate_nhanes_inputs,
    predict_nhanes_screening,
)
from app.utils.model_loader import load_pipeline, load_model_metadata

logger = logging.getLogger(__name__)


def render_nhanes_page() -> None:
    """Renders the complete interactive NHANES screening interface."""
    st.markdown(
        """
        <div style="padding-bottom: 5px;">
            <h2 style="color: #1E3A8A; margin-bottom: 2px;">🧬 NHANES Non-Laboratory Cholesterol Screening</h2>
            <p style="color: #4B5563; font-size: 0.95rem; margin-top: 0;">
                Pre-diagnostic community screening engine evaluating 24 non-invasive demographic, anthropometric,
                vital sign, and lifestyle predictors to assess screening likelihood of phenotypic hypercholesterolemia.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Academic Disclaimer Banner
    st.markdown(
        """
        <div style="background-color: #FEF3C7; border-left: 4px solid #F59E0B; padding: 10px 14px;
                    border-radius: 4px; margin-bottom: 16px; font-size: 0.82rem; color: #92400E;">
            <strong>Academic Research Prototype Notice:</strong>
            This tool is developed as part of a final-year BTech capstone project.
            It generates an estimated screening score based on retrospective NHANES 2021–2023 survey data.
            It does <em>not</em> measure serum cholesterol, does <em>not</em> provide a medical diagnosis,
            and must <em>never</em> replace laboratory blood analysis or clinical consultation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Check Model Health Before Displaying Form
    pipeline = load_pipeline("nhanes_screening_core")
    metadata = load_model_metadata()

    if pipeline is None:
        st.error(
            "⚠️ **Screening Engine Unavailable:** The pre-trained NHANES core model (`models/nhanes_screening_core.joblib`) "
            "could not be loaded. Please ensure model artifacts are present in the repository.",
            icon="🚨",
        )
        return

    # Extract calibrated operating threshold from metadata
    core_meta = metadata.get("artifacts", {}).get("nhanes_screening_core", {})
    operating_threshold = float(core_meta.get("selected_decision_threshold", 0.4600))

    # Reset Action
    col_t1, col_t2 = st.columns([4, 1])
    with col_t2:
        if st.button("🔄 Reset to Defaults", help="Reset all form inputs to standard population median values"):
            for key in list(st.session_state.keys()):
                if key.startswith("nhanes_"):
                    del st.session_state[key]
            st.rerun()

    # 3. Interactive Patient Screening Form
    with st.form("nhanes_screening_form"):
        st.markdown("#### 1. Demographic Profile")
        c_dem1, c_dem2, c_dem3 = st.columns(3)

        with c_dem1:
            age_years = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=100,
                value=st.session_state.get("nhanes_age", 52),
                step=1,
                help="Adults aged 18 to 80+ surveyed in NHANES.",
                key="nhanes_age",
            )
            sex_choice = st.selectbox(
                "Biological Sex",
                options=["Male", "Female"],
                index=0 if st.session_state.get("nhanes_sex", "Male") == "Male" else 1,
                help="Biological sex as recorded in the examination center.",
                key="nhanes_sex",
            )

        with c_dem2:
            race_choice = st.selectbox(
                "Race / Ethnicity",
                options=VALID_RACE_ETHNICITY_CATEGORIES,
                index=VALID_RACE_ETHNICITY_CATEGORIES.index(
                    st.session_state.get("nhanes_race", "Non-Hispanic White")
                ),
                help="Self-reported demographic group matching CDC NHANES categories.",
                key="nhanes_race",
            )
            education_choice = st.selectbox(
                "Educational Attainment",
                options=VALID_EDUCATION_CATEGORIES,
                index=VALID_EDUCATION_CATEGORIES.index(
                    st.session_state.get("nhanes_edu", "College graduate or above")
                ),
                help="Highest completed level of education.",
                key="nhanes_edu",
            )

        with c_dem3:
            poverty_ratio = st.number_input(
                "Family Income-to-Poverty Ratio (PIR)",
                min_value=0.0,
                max_value=5.0,
                value=float(st.session_state.get("nhanes_pir", 3.2)),
                step=0.1,
                format="%.1f",
                help="Ratio of family income to federal poverty threshold (1.0 = poverty level; 5.0 = 500%+ of poverty line).",
                key="nhanes_pir",
            )

        st.markdown("---")
        st.markdown("#### 2. Anthropometric Measurements")
        c_ant1, c_ant2, c_ant3 = st.columns(3)

        with c_ant1:
            standing_height = st.number_input(
                "Standing Height (cm)",
                min_value=110.0,
                max_value=230.0,
                value=float(st.session_state.get("nhanes_ht", 172.0)),
                step=0.5,
                help="Used to derive body mass index and waist-to-height ratio.",
                key="nhanes_ht",
            )
            weight_kg = st.number_input(
                "Body Weight (kg)",
                min_value=30.0,
                max_value=250.0,
                value=float(st.session_state.get("nhanes_wt", 82.0)),
                step=0.5,
                help="Current body weight measured in light clothing.",
                key="nhanes_wt",
            )

        # Derived BMI calculation helper
        computed_bmi = round(weight_kg / ((standing_height / 100.0) ** 2), 2)

        with c_ant2:
            bmi_input = st.number_input(
                "Body Mass Index (BMI, kg/m²)",
                min_value=12.0,
                max_value=75.0,
                value=float(st.session_state.get("nhanes_bmi", computed_bmi)),
                step=0.1,
                format="%.1f",
                help=f"Auto-calculated from height & weight ({computed_bmi} kg/m²), or can be adjusted.",
                key="nhanes_bmi",
            )
            waist_cm = st.number_input(
                "Waist Circumference (cm)",
                min_value=45.0,
                max_value=200.0,
                value=float(st.session_state.get("nhanes_waist", 96.0)),
                step=0.5,
                help="Measured at the uppermost lateral border of the right ilium.",
                key="nhanes_waist",
            )

        # Derived waist-to-height ratio
        computed_wthr = round(waist_cm / standing_height, 4)

        with c_ant3:
            st.metric("Waist-to-Height Ratio", f"{computed_wthr:.3f}", help="Auto-derived: Waist (cm) / Height (cm)")
            wt_change_1yr = st.number_input(
                "1-Year Weight Change (kg)",
                min_value=-50.0,
                max_value=50.0,
                value=float(st.session_state.get("nhanes_wtchg", 0.0)),
                step=0.5,
                help="Net weight change compared to 12 months ago (0 = stable, negative = lost, positive = gained).",
                key="nhanes_wtchg",
            )

        st.markdown("---")
        st.markdown("#### 3. Vital Signs & Hemodynamics")
        c_vit1, c_vit2, c_vit3 = st.columns(3)

        with c_vit1:
            systolic_bp = st.number_input(
                "Systolic Blood Pressure (mmHg)",
                min_value=70.0,
                max_value=240.0,
                value=float(st.session_state.get("nhanes_sys", 126.0)),
                step=1.0,
                help="Resting average seated systolic blood pressure.",
                key="nhanes_sys",
            )
            diastolic_bp = st.number_input(
                "Diastolic Blood Pressure (mmHg)",
                min_value=35.0,
                max_value=140.0,
                value=float(st.session_state.get("nhanes_dia", 80.0)),
                step=1.0,
                help="Resting average seated diastolic blood pressure.",
                key="nhanes_dia",
            )

        computed_pp = max(0.0, round(systolic_bp - diastolic_bp, 1))

        with c_vit2:
            st.metric("Pulse Pressure (mmHg)", f"{computed_pp:.1f}", help="Auto-derived: Systolic BP - Diastolic BP")
            resting_pulse = st.number_input(
                "Resting Heart Rate (bpm)",
                min_value=35.0,
                max_value=180.0,
                value=float(st.session_state.get("nhanes_pulse", 72.0)),
                step=1.0,
                help="Resting radial pulse in beats per minute.",
                key="nhanes_pulse",
            )

        with c_vit3:
            bp_med_choice = st.radio(
                "Prescribed Blood Pressure Medication",
                options=["No", "Yes"],
                index=None,
                help="Currently taking medication prescribed by a physician for high blood pressure.",
                key="nhanes_bpmed",
            )

        st.markdown("---")
        st.markdown("#### 4. Lifestyle & Behavioral Profile")
        c_life1, c_life2 = st.columns(2)

        with c_life1:
            smoking_choice = st.selectbox(
                "Smoking Status",
                options=[
                    "Never (smoked < 100 cigarettes in lifetime)",
                    "Former (smoked >= 100 cigarettes but quit)",
                    "Current smoker",
                ],
                index=0,
                help="Tobacco cigarette smoking exposure.",
                key="nhanes_smoke",
            )
            ltp_mets = st.number_input(
                "Leisure Physical Activity (MET-min/week)",
                min_value=0.0,
                max_value=15000.0,
                value=float(st.session_state.get("nhanes_mets", 480.0)),
                step=30.0,
                help="Moderate/vigorous leisure activity (e.g., 150 min moderate walk ≈ 500-600 MET-min/week).",
                key="nhanes_mets",
            )
            sedentary_mins = st.number_input(
                "Sedentary Sitting Time (minutes/day)",
                min_value=0.0,
                max_value=1440.0,
                value=float(st.session_state.get("nhanes_sed", 360.0)),
                step=15.0,
                help="Typical daily minutes spent sitting at work, home, or commuting.",
                key="nhanes_sed",
            )

        with c_life2:
            alcohol_drinks = st.number_input(
                "Alcohol Consumption (drinks/week)",
                min_value=0.0,
                max_value=80.0,
                value=float(st.session_state.get("nhanes_alc", 2.0)),
                step=1.0,
                help="Average standard alcoholic drinks consumed per week.",
                key="nhanes_alc",
            )
            sleep_hours = st.number_input(
                "Weekday Sleep Duration (hours/night)",
                min_value=2.0,
                max_value=16.0,
                value=float(st.session_state.get("nhanes_sleep", 7.5)),
                step=0.5,
                help="Average self-reported hours of sleep on weeknights.",
                key="nhanes_sleep",
            )

        st.markdown("---")
        st.markdown("#### 5. Medical History & Symptoms")
        c_med1, c_med2 = st.columns(2)

        with c_med1:
            diabetes_choice = st.radio(
                "Diagnosed Diabetes (Doctor-Reported)",
                options=["No", "Yes"],
                index=None,
                help="Ever informed by a doctor or health professional that you have diabetes.",
                key="nhanes_diab",
            )
            gen_health_choice = st.selectbox(
                "Self-Rated General Health",
                options=list(VALID_GENERAL_HEALTH_CHOICES.keys()),
                index=1,  # "Very good" default
                help="Subjective overall self-assessment of health.",
                key="nhanes_gh",
            )
            phq9_score = st.number_input(
                "PHQ-9 Depression Screener Score (0 to 27)",
                min_value=0.0,
                max_value=27.0,
                value=float(st.session_state.get("nhanes_phq9", 2.0)),
                step=1.0,
                help="Patient Health Questionnaire-9 total score (0-4: minimal, 5-9: mild, 10+: moderate/severe).",
                key="nhanes_phq9",
            )

        with c_med2:
            chd_choice = st.radio(
                "Coronary Heart Disease (CHD) History",
                options=["No", "Yes"],
                index=None,
                help="Ever diagnosed with coronary heart disease.",
                key="nhanes_chd",
            )
            stroke_choice = st.radio(
                "Stroke History",
                options=["No", "Yes"],
                index=None,
                help="Ever diagnosed with a stroke.",
                key="nhanes_stroke",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔍 Run Non-Laboratory Screening Assessment",
            type="primary",
            use_container_width=True,
        )

    # 4. Form Processing & Inference Lifecycle
    if submitted:
        # Map human-readable selections to exact schema
        smoke_map = {
            "Never (smoked < 100 cigarettes in lifetime)": "never",
            "Former (smoked >= 100 cigarettes but quit)": "former",
            "Current smoker": "current",
        }

        form_payload: Dict[str, Any] = {
            "age_years": age_years,
            "sex": sex_choice.lower(),
            "race_ethnicity": race_choice,
            "education_level": education_choice,
            "family_income_poverty_ratio": poverty_ratio,
            "bmi_kg_m2": bmi_input,
            "waist_circumference_cm": waist_cm,
            "height_cm": standing_height,
            "waist_to_height_ratio": computed_wthr,
            "weight_change_1yr_kg": wt_change_1yr,
            "avg_systolic_bp_mmhg": systolic_bp,
            "avg_diastolic_bp_mmhg": diastolic_bp,
            "pulse_pressure_mmhg": computed_pp,
            "avg_pulse_bpm": resting_pulse,
            "bp_medication": 1.0 if bp_med_choice == "Yes" else (0.0 if bp_med_choice == "No" else None),
            "smoking_status": smoke_map.get(smoking_choice, "never"),
            "total_ltp_met_minutes_week": ltp_mets,
            "sedentary_minutes_day": sedentary_mins,
            "alcohol_drinks_per_week": alcohol_drinks,
            "sleep_hours_weekday": sleep_hours,
            "diabetes_diagnosed": 1.0 if diabetes_choice == "Yes" else (0.0 if diabetes_choice == "No" else None),
            "general_health": VALID_GENERAL_HEALTH_CHOICES.get(gen_health_choice, 3.0),
            "phq9_score": phq9_score,
            "coronary_heart_disease": 1.0 if chd_choice == "Yes" else (0.0 if chd_choice == "No" else None),
            "stroke": 1.0 if stroke_choice == "Yes" else (0.0 if stroke_choice == "No" else None),
        }

        # Step 3: Input Validation
        is_valid, validation_errors, df_validated = validate_nhanes_inputs(form_payload)

        if not is_valid or df_validated is None:
            st.error("### ⚠️ Input Validation Failed", icon="🚫")
            for err in validation_errors:
                st.markdown(f"- {err}")
            return

        # Step 4: Model Inference
        try:
            with st.spinner("Analyzing non-laboratory screening biomarkers..."):
                result = predict_nhanes_screening(df_validated)
        except Exception as exc:
            logger.exception("Prediction failed during screening")
            st.error(
                f"**Assessment Error:** An unexpected error occurred during prediction ({type(exc).__name__}). "
                "Please verify your inputs or contact system maintenance.",
                icon="❌",
            )
            return

        # Step 5: Render Results Interface
        st.markdown("---")
        st.markdown("### 📋 Screening Assessment Results")

        score_pct = result["screening_score_pct"]
        threshold_pct = result["operating_threshold_pct"]
        is_higher = result["is_higher_signal"]

        # 1. Main Classification Card
        if is_higher:
            banner_bg = "#FEF2F2"
            banner_border = "#EF4444"
            banner_text = "#B91C1C"
            badge_title = "HIGHER SCREENING SIGNAL DETECTED"
            summary_statement = (
                f"The model estimated a screening score of **{score_pct:.1f}%**, which meets or exceeds "
                f"the calibrated sensitivity operating threshold of **{threshold_pct:.1f}%**."
            )
        else:
            banner_bg = "#F0FDF4"
            banner_border = "#22C55E"
            banner_text = "#15803D"
            badge_title = "LOWER SCREENING SIGNAL"
            summary_statement = (
                f"The model estimated a screening score of **{score_pct:.1f}%**, which is below "
                f"the calibrated sensitivity operating threshold of **{threshold_pct:.1f}%**."
            )

        st.markdown(
            f"""
            <div style="background-color: {banner_bg}; border: 2px solid {banner_border}; border-radius: 8px;
                        padding: 16px 20px; margin-bottom: 20px;">
                <div style="font-size: 1.25rem; font-weight: 800; color: {banner_text}; margin-bottom: 6px;">
                    {badge_title}
                </div>
                <div style="font-size: 0.95rem; color: #1F2937; line-height: 1.45;">
                    {summary_statement}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Score & Operating Threshold Breakdown
        c_res1, c_res2, c_res3, c_res4 = st.columns(4)
        c_res1.metric("Screening Score", f"{score_pct:.1f}%")
        c_res2.metric("Operating Threshold", f"{threshold_pct:.1f}%", help="Threshold calibrated on training data targeting ~80% sensitivity")
        c_res3.metric("Held-Out Sensitivity", f"{result['calibrated_test_sensitivity']*100:.1f}%", help="Observed sensitivity on independent test cohort")
        c_res4.metric("Held-Out Specificity", f"{result['calibrated_test_specificity']*100:.1f}%", help="Observed specificity on independent test cohort")

        # Visual Score Gauge Bar
        st.markdown("**Screening Score Relative to Calibrated Threshold:**")
        st.progress(min(1.0, max(0.0, score_pct / 100.0)))
        st.caption(
            f"Marker: {score_pct:.1f}% | Calibrated Operating Cutoff: {threshold_pct:.1f}% "
            f"({result['classification_label']})"
        )

        # 3. Clinical Interpretation & Next-Step Guidance
        st.markdown("#### Clinical Interpretation & Recommended Actions")

        if is_higher:
            st.markdown(
                r"""
                - **What this means:** The submitted non-laboratory profile exhibits biometric, metabolic, and hemodynamic
                  patterns commonly observed among individuals with phenotypic hypercholesterolemia (Total Cholesterol $\ge 240$ mg/dL
                  or statin pharmacotherapy) in the US population.
                - **What this does NOT mean:** This is **not a clinical diagnosis** of hypercholesterolemia. Elevated screening scores
                  can occur due to overlapping risk factors (such as elevated blood pressure, central adiposity, or age) without elevated cholesterol.
                - **Recommended Action:**
                  1. Schedule an appointment with a primary care physician.
                  2. Obtain a standard fasting lipid panel (Total Cholesterol, LDL-C, HDL-C, Triglycerides).
                  3. Discuss comprehensive cardiovascular health management.
                """
            )
        else:
            st.markdown(
                r"""
                - **What this means:** The submitted profile exhibits biometric patterns less characteristic of individuals
                  with severe phenotypic hypercholesterolemia.
                - **What this does NOT mean:** This does **not guarantee** normal blood cholesterol levels and does not rule out
                  atherosclerotic cardiovascular risk. Routine lipid screening remains essential.
                - **Recommended Action:** Maintain regular preventive health check-ups and follow standard periodic lipid testing guidelines.
                """
            )

        # 4. Academic Limitations & Model Governance
        with st.expander("Academic Methodology & Model Governance Notice", expanded=False):
            st.markdown(
                rf"""
                - **Dataset Provenance:** Model trained on CDC NHANES 2021–2023 examination survey cohort ($N = 5,774$ evaluable participants).
                - **Target Definition:** `{result['target_definition']}` (Option B proxy).
                - **Estimator Architecture:** Pre-trained Scikit-Learn `{result['estimator_class']}` pipeline (`models/nhanes_screening_core.joblib`).
                - **Threshold Derivation:** The operating threshold ($\tau = {result['operating_threshold']:.2f}$) was selected strictly from
                  training-set 5-fold cross-validation out-of-fold predictions to prioritize screening sensitivity ($\ge 80\%$).
                  On the quarantined held-out test cohort ($N = 1,155$), observed sensitivity was **{result['calibrated_test_sensitivity']*100:.1f}%**
                  and specificity was **{result['calibrated_test_specificity']*100:.1f}%** (ROC-AUC: **{result['calibrated_test_roc_auc']:.4f}**).
                - **Non-Diagnostic Scope:** This model is an academic research prototype. It does not predict future 10-year cardiovascular events
                  and has not been cleared for individual medical diagnosis.
                """
            )

        # 5. Submitted Inputs Summary
        with st.expander("Review Submitted Feature Vector (24 Core Predictors)", expanded=False):
            summary_df = pd.DataFrame(
                [
                    {"Predictor Column": col, "Submitted Value": str(result["input_summary"][col])}
                    for col in NHANES_CORE_PREDICTORS
                ]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
