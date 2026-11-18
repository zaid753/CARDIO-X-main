"""
CARDIO-X: Interactive Framingham 10-Year CVD Risk Prediction Module.

Implements the user-facing clinical risk form, physiological validation, model prediction,
score presentation, derived hemodynamics, clinical next-step guidance, and academic governance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import streamlit as st
import pandas as pd

from app.utils.framingham_engine import (
    FRAMINGHAM_CORE_PREDICTORS,
    VALID_SEX_CHOICES,
    VALID_EDUC_CHOICES,
    validate_framingham_inputs,
    predict_framingham_risk,
)
from app.utils.model_loader import load_pipeline, load_model_metadata

logger = logging.getLogger(__name__)


def render_framingham_page() -> None:
    """Renders the complete interactive Framingham 10-year CVD risk interface."""
    st.markdown(
        """
        <div style="padding-bottom: 5px;">
            <h2 style="color: #1E3A8A; margin-bottom: 2px;">⏱️ Framingham 10-Year Cardiovascular Disease Risk</h2>
            <p style="color: #4B5563; font-size: 0.95rem; margin-top: 0;">
                Primary prevention risk estimation predicting first incident hard cardiovascular disease event within 10 years
                (3,652.5 days) evaluating 14 baseline clinical, demographic, and laboratory biomarkers.
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
            This tool is developed as part of an academic capstone informatics project utilizing retrospective Framingham Heart Study cohort data.
            It is a <em>conditional landmark binary classifier</em> and is <strong>not</strong> a competing-risk survival model,
            an FDA-cleared medical device, or an official replacement for clinical risk calculators (such as the ACC/AHA Pooled Cohort Equations).
            It must <em>never</em> be used to guide individual medical interventions, prescribe medications, or replace direct physician consultation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Check Model Health Before Displaying Form
    pipeline = load_pipeline("framingham_risk")
    metadata = load_model_metadata()

    if pipeline is None:
        st.error(
            "⚠️ **Risk Engine Unavailable:** The pre-trained Framingham risk model (`models/framingham_risk.joblib`) "
            "could not be loaded. Please verify that model artifacts are present in the repository.",
            icon="🚨",
        )
        return

    # Extract calibrated operating threshold from metadata
    fram_meta = metadata.get("artifacts", {}).get("framingham_risk", {})
    raw_threshold = fram_meta.get("selected_decision_threshold")
    operating_threshold = float(raw_threshold) if raw_threshold is not None else 0.4000

    # 3. Patient Clinical Input Form
    with st.form("framingham_risk_form"):
        st.markdown("### Patient Baseline Clinical Covariates")
        st.caption(
            "Enter baseline demographic, hemodynamic, lifestyle, and laboratory markers. "
            "All fields marked with an asterisk (*) are required."
        )

        # ---------------- Section 1: Demographics ----------------
        st.markdown("#### 1. Demographics & Educational Attainment")
        c_dem1, c_dem2, c_dem3 = st.columns(3)

        with c_dem1:
            age_years = st.number_input(
                "Patient Age (years) *",
                min_value=30.0,
                max_value=85.0,
                value=float(st.session_state.get("fram_age", 52.0)),
                step=1.0,
                help="Age in years at baseline examination (allowable range: 30 to 85).",
                key="fram_age",
            )

        with c_dem2:
            sex_choice = st.radio(
                "Biological Sex *",
                options=list(VALID_SEX_CHOICES.keys()),
                index=None,
                help="Biological sex at baseline examination (Female = 0, Male = 1).",
                key="fram_sex",
            )

        with c_dem3:
            educ_choice = st.selectbox(
                "Educational Attainment (educ)",
                options=list(VALID_EDUC_CHOICES.keys()),
                index=1,  # Default to High School Diploma / GED
                help="Highest completed level of formal schooling.",
                key="fram_educ",
            )

        st.markdown("---")

        # ---------------- Section 2: Blood Pressure & Hemodynamics ----------------
        st.markdown("#### 2. Blood Pressure & Hemodynamic Parameters")
        c_bp1, c_bp2, c_bp3 = st.columns(3)

        with c_bp1:
            systolic_bp = st.number_input(
                "Systolic Blood Pressure (SYSBP, mmHg) *",
                min_value=70.0,
                max_value=240.0,
                value=float(st.session_state.get("fram_sysbp", 132.0)),
                step=1.0,
                help="Resting average seated systolic blood pressure in mmHg.",
                key="fram_sysbp",
            )
            diastolic_bp = st.number_input(
                "Diastolic Blood Pressure (DIABP, mmHg) *",
                min_value=40.0,
                max_value=140.0,
                value=float(st.session_state.get("fram_diabp", 84.0)),
                step=1.0,
                help="Resting average seated diastolic blood pressure in mmHg.",
                key="fram_diabp",
            )

        # Pre-compute hemodynamic markers
        pp_display = max(0.0, round(systolic_bp - diastolic_bp, 1))
        map_display = round(diastolic_bp + (pp_display / 3.0), 1)

        with c_bp2:
            st.metric(
                "Pulse Pressure (mmHg)",
                f"{pp_display:.1f}",
                help="Derived: Systolic BP - Diastolic BP (marker of large artery stiffness)",
            )
            st.metric(
                "Mean Arterial Pressure (mmHg)",
                f"{map_display:.1f}",
                help="Derived: Diastolic BP + (Pulse Pressure / 3) (marker of steady-state perfusion)",
            )

        with c_bp3:
            bp_meds_choice = st.radio(
                "Antihypertensive Medication (BPMEDS) *",
                options=["No", "Yes"],
                index=None,
                help="Currently taking physician-prescribed blood pressure medication.",
                key="fram_bpmeds",
            )
            prev_hyp_choice = st.radio(
                "Prevalent Hypertension History (PREVHYP) *",
                options=["No", "Yes"],
                index=None,
                help="Prior clinical diagnosis or documentation of hypertension.",
                key="fram_prevhyp",
            )

        st.markdown("---")

        # ---------------- Section 3: Smoking History ----------------
        st.markdown("#### 3. Tobacco Smoking Profile")
        c_smk1, c_smk2 = st.columns(2)

        with c_smk1:
            cur_smoke_choice = st.radio(
                "Current Cigarette Smoker (CURSMOKE) *",
                options=["No", "Yes"],
                index=None,
                help="Currently smoking cigarettes regularly at baseline examination.",
                key="fram_cursmoke",
            )

        with c_smk2:
            cig_per_day = st.number_input(
                "Cigarettes Per Day (CIGPDAY) *",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get("fram_cigpday", 0.0)),
                step=1.0,
                help="Average number of cigarettes smoked per day (0 for non-smokers).",
                key="fram_cigpday",
            )

        st.markdown("---")

        # ---------------- Section 4: Anthropometrics & Vital Signs ----------------
        st.markdown("#### 4. Anthropometrics & Resting Pulse")
        c_ant1, c_ant2, c_ant3 = st.columns(3)

        with c_ant1:
            height_cm = st.number_input(
                "Standing Height (cm)",
                min_value=120.0,
                max_value=230.0,
                value=float(st.session_state.get("fram_ht", 170.0)),
                step=0.5,
                help="Height used to calculate Body Mass Index.",
                key="fram_ht",
            )
            weight_kg = st.number_input(
                "Body Weight (kg)",
                min_value=30.0,
                max_value=250.0,
                value=float(st.session_state.get("fram_wt", 75.0)),
                step=0.5,
                help="Weight used to calculate Body Mass Index.",
                key="fram_wt",
            )

        computed_bmi = round(weight_kg / ((height_cm / 100.0) ** 2), 2)

        with c_ant2:
            bmi_input = st.number_input(
                "Body Mass Index (BMI, kg/m²)",
                min_value=12.0,
                max_value=70.0,
                value=float(st.session_state.get("fram_bmi", computed_bmi)),
                step=0.1,
                format="%.1f",
                help=f"Auto-calculated from height & weight ({computed_bmi} kg/m²), or can be entered directly.",
                key="fram_bmi",
            )

        with c_ant3:
            heart_rate = st.number_input(
                "Resting Heart Rate (HEARTRTE, bpm)",
                min_value=35.0,
                max_value=180.0,
                value=float(st.session_state.get("fram_hr", 72.0)),
                step=1.0,
                help="Resting ventricular rate in beats per minute.",
                key="fram_hr",
            )

        st.markdown("---")

        # ---------------- Section 5: Laboratory Biomarkers & Glycemic Status ----------------
        st.markdown("#### 5. Baseline Laboratory Biomarkers & Diabetes Status")
        c_lab1, c_lab2, c_lab3 = st.columns(3)

        with c_lab1:
            tot_chol = st.number_input(
                "Total Serum Cholesterol (TOTCHOL, mg/dL) *",
                min_value=100.0,
                max_value=600.0,
                value=float(st.session_state.get("fram_totchol", 215.0)),
                step=1.0,
                help="Baseline venous blood total cholesterol in mg/dL.",
                key="fram_totchol",
            )

        with c_lab2:
            blood_glucose = st.number_input(
                "Casual/Fasting Blood Glucose (GLUCOSE, mg/dL)",
                min_value=40.0,
                max_value=500.0,
                value=float(st.session_state.get("fram_glucose", 82.0)),
                step=1.0,
                help="Baseline serum glucose in mg/dL (missing values imputed via cohort median).",
                key="fram_glucose",
            )

        with c_lab3:
            diabetes_choice = st.radio(
                "Diagnosed Diabetes (DIABETES) *",
                options=["No", "Yes"],
                index=None,
                help="History of diagnosed diabetes mellitus at baseline.",
                key="fram_diabetes",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "⚡ Compute 10-Year Incident CVD Risk",
            type="primary",
            use_container_width=True,
        )

    # 4. Form Processing & Inference Lifecycle
    if submitted:
        # Explicit mapping for binary choices without silent defaulting
        def _map_binary(choice: Optional[str]) -> Optional[float]:
            if choice == "Yes":
                return 1.0
            if choice == "No":
                return 0.0
            return None

        # Build raw form payload
        form_payload: Dict[str, Any] = {
            "AGE": age_years,
            "SEX": VALID_SEX_CHOICES.get(sex_choice) if sex_choice else None,
            "CURSMOKE": _map_binary(cur_smoke_choice),
            "CIGPDAY": cig_per_day,
            "educ": VALID_EDUC_CHOICES.get(educ_choice, 2.0),
            "BPMEDS": _map_binary(bp_meds_choice),
            "PREVHYP": _map_binary(prev_hyp_choice),
            "HEARTRTE": heart_rate,
            "TOTCHOL": tot_chol,
            "BMI": bmi_input,
            "DIABETES": _map_binary(diabetes_choice),
            "GLUCOSE": blood_glucose,
            "SYSBP": systolic_bp,
            "DIABP": diastolic_bp,
        }

        # Step 3: Validate Inputs
        is_valid, validation_errors, df_validated = validate_framingham_inputs(form_payload)

        if not is_valid or df_validated is None:
            st.error("### ⚠️ Input Validation Failed", icon="🚫")
            for err in validation_errors:
                st.markdown(f"- {err}")
            return

        # Step 4: Model Inference
        try:
            with st.spinner("Computing 10-year incident cardiovascular disease risk projection..."):
                result = predict_framingham_risk(df_validated)
        except Exception as exc:
            logger.exception("Prediction failed during Framingham risk calculation")
            st.error(
                f"🚨 **Risk Calculation Error:** An unexpected error occurred during model evaluation: `{exc}`",
                icon="⚠️",
            )
            return

        # 5. Results Presentation
        st.markdown("---")
        st.markdown("### 📊 Framingham 10-Year CVD Risk Estimation Results")

        score_pct = result["risk_score_pct"]
        threshold_pct = result["operating_threshold_pct"]
        is_higher = result["is_higher_signal"]

        # Classification Banner
        if is_higher:
            st.markdown(
                f"""
                <div style="background-color: #FEF2F2; border-left: 6px solid #EF4444; padding: 14px 18px;
                            border-radius: 6px; margin-bottom: 20px;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #991B1B;">
                        ⚠️ {result['classification_label']}
                    </div>
                    <div style="font-size: 0.95rem; color: #7F1D1D; margin-top: 4px;">
                        The submitted risk factor profile produces an estimated 10-year incident CVD score of
                        <strong>{score_pct:.1f}%</strong>, which meets or exceeds the calibrated sensitivity threshold
                        of <strong>{threshold_pct:.1f}%</strong>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background-color: #ECFDF5; border-left: 6px solid #10B981; padding: 14px 18px;
                            border-radius: 6px; margin-bottom: 20px;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #065F46;">
                        ✅ {result['classification_label']}
                    </div>
                    <div style="font-size: 0.95rem; color: #064E3B; margin-top: 4px;">
                        The submitted risk factor profile produces an estimated 10-year incident CVD score of
                        <strong>{score_pct:.1f}%</strong>, which remains below the calibrated sensitivity threshold
                        of <strong>{threshold_pct:.1f}%</strong>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Metric Summary Row
        c_res1, c_res2, c_res3, c_res4 = st.columns(4)
        c_res1.metric(
            "Estimated 10-Year Score",
            f"{score_pct:.1f}%",
            delta=f"{score_pct - threshold_pct:+.1f}% vs threshold",
            delta_color="inverse",
            help="Model positive-class pseudo-probability for first incident hard CVD within 10 years",
        )
        c_res2.metric(
            "Operating Threshold (τ*)",
            f"{threshold_pct:.1f}%",
            help="Calibrated threshold targeting >= 80% sensitivity in primary prevention cohort",
        )
        c_res3.metric(
            "Test Sensitivity (τ*=0.40)",
            f"{result['calibrated_test_sensitivity']*100:.1f}%",
            help="Observed sensitivity on quarantined Framingham test cohort (74 events)",
        )
        c_res4.metric(
            "Test Specificity (τ*=0.40)",
            f"{result['calibrated_test_specificity']*100:.1f}%",
            help="Observed specificity on quarantined Framingham test cohort",
        )

        # Visual Score Gauge Bar
        st.markdown("**10-Year CVD Risk Score Relative to Operating Threshold:**")
        st.progress(min(1.0, max(0.0, score_pct / 100.0)))
        st.caption(
            f"Estimated Score: {score_pct:.1f}% | Operating Threshold: {threshold_pct:.1f}% "
            f"({result['classification_label']})"
        )

        # Derived Hemodynamics Card
        hemo = result.get("derived_hemodynamics", {})
        c_h1, c_h2 = st.columns(2)
        c_h1.info(f"**Derived Pulse Pressure:** `{hemo.get('pulse_pressure_mmhg', 0.0)} mmHg` (SYSBP - DIABP)")
        c_h2.info(f"**Derived Mean Arterial Pressure (MAP):** `{hemo.get('mean_arterial_pressure_mmhg', 0.0)} mmHg` (DIABP + PP/3)")

        # 6. Clinical Interpretation & Next-Step Guidance
        st.markdown("#### Clinical Interpretation & Recommended Actions")

        if is_higher:
            st.markdown(
                r"""
                - **What this means:** The submitted clinical covariate profile exhibits a constellation of vascular,
                  metabolic, and demographic risk factors (e.g., elevated blood pressure, dyslipidemia, smoking, or age)
                  statistically associated with a higher likelihood of experiencing a first incident hard cardiovascular event
                  (myocardial infarction, coronary death, or stroke) over a 10-year follow-up horizon.
                - **What this does NOT mean:** This score is **not an individualized clinical certainty**. It does not diagnose
                  current coronary artery disease or guarantee a future adverse event.
                - **Recommended Actions:**
                  1. Consult a board-certified primary care physician or cardiologist for comprehensive clinical cardiovascular risk evaluation.
                  2. Obtain complete fasting lipid profile testing (Total Cholesterol, LDL-C, HDL-C, Triglycerides).
                  3. Discuss clinically validated guideline risk estimators (such as the ACC/AHA Pooled Cohort Equations) and appropriate lifestyle or pharmacological primary-prevention interventions.
                """
            )
        else:
            st.markdown(
                r"""
                - **What this means:** The submitted profile exhibits lower statistical resemblance to Framingham participants
                  who developed incident cardiovascular disease within 10 years under baseline surveillance.
                - **What this does NOT mean:** This does **not imply zero risk** and does not guarantee immunity from cardiovascular disease.
                  Cardiovascular risk increases progressively with age and cumulative atherogenic exposure.
                - **Recommended Actions:** Maintain routine periodic clinical examinations, monitor blood pressure, adhere to a heart-healthy diet, and remain physically active.
                """
            )

        # 7. Academic Methodology & Model Governance Drawer
        with st.expander("Academic Methodology & Model Governance Notice", expanded=False):
            st.markdown(
                rf"""
                - **Dataset Provenance:** Framingham Heart Study examination cohort ($N = 3,983$ evaluable primary prevention participants).
                - **Target Definition:** `{result['target_definition']}` (`TARGET_10Y_CVD`).
                - **Model Architecture:** Pre-trained Scikit-Learn L2-regularized `{result['estimator_class']}` (`models/framingham_risk.joblib`)
                  trained on the Standard BP feature configuration with `RobustScaler` applied to continuous variables.
                - **Operating Threshold:** The decision cutoff ($\tau^* = {result['operating_threshold']:.2f}$) was calibrated strictly on
                  training-set 5-fold cross-validation out-of-fold predictions to prevent sensitivity collapse and achieve $\ge 80\%$ recall.
                - **Held-Out Test Performance ($N = 797$, 74 events):**
                  - Test ROC-AUC: **{result['calibrated_test_roc_auc']:.4f}**
                  - Test Sensitivity: **{result['calibrated_test_sensitivity']*100:.1f}%**
                  - Test Specificity: **{result['calibrated_test_specificity']*100:.1f}%**
                  - Test Accuracy: **{result['calibrated_test_accuracy']*100:.1f}%**
                  - Test F1 Score: **{result['calibrated_test_f1']:.4f}**
                - **Methodological Limitations:**
                  1. Formulated as a conditional landmark binary classifier on participants with complete follow-up. It is **not** a competing-risk survival model.
                  2. Excludes individuals with baseline coronary disease, stroke, or transient ischemic attack.
                  3. This is an academic research prototype and has **not** been cleared by regulatory authorities for clinical medical decision-making.
                """
            )

        # 8. Submitted Features Summary
        with st.expander("Review Submitted Feature Vector (14 Baseline Predictors)", expanded=False):
            summary_df = pd.DataFrame(
                [
                    {"Predictor Column": col, "Submitted Value": str(result["input_summary"][col])}
                    for col in FRAMINGHAM_CORE_PREDICTORS
                ]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
