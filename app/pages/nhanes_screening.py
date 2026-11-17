"""
CARDIO-X NHANES Cholesterol Screening Module (Placeholder).
"""

import streamlit as st


def render_nhanes_page() -> None:
    """Renders the NHANES Non-Laboratory Screening Module placeholder."""
    st.markdown("### 🧬 NHANES Non-Laboratory Cholesterol Screening")
    st.markdown(
        """
        *Pre-diagnostic community screening engine predicting phenotypic hypercholesterolemia*
        `TARGET_HYPERCHOL = (Total Cholesterol >= 240 mg/dL OR Lipid Medication == 1)`
        """
    )

    st.markdown("---")

    st.info(
        "🚧 **Module Under Active Development (Scheduled for Phase 3B)**\n\n"
        "The interactive patient screening form will accept 24 non-invasive clinical predictors:\n"
        "- **Demographics:** Age, sex, race/ethnicity, education, poverty ratio.\n"
        "- **Anthropometrics:** BMI, waist circumference, waist-to-height ratio, 1-year weight change.\n"
        "- **Vitals & Hemodynamics:** Systolic BP, diastolic BP, pulse pressure, resting pulse, BP medication.\n"
        "- **Lifestyle Habits:** Smoking history, physical activity METs, sedentary time, alcohol consumption, sleep hours.\n"
        "- **Medical History:** Diagnosed diabetes, general health score, PHQ-9 depression score, CHD history, stroke history.\n\n"
        "Predictions will be computed using the pre-trained `models/nhanes_screening_core.joblib` pipeline at the calibrated threshold $\\tau^* = 0.46$.",
        icon="🛠️"
    )

    st.markdown("#### Model Specification Preview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predictor Count", "24 Features", "Non-Laboratory")
    c2.metric("Target Sensitivity", "77.9%", "High-Recall Screening")
    c3.metric("Operating Threshold", "0.46", "Calibrated for Triage")
