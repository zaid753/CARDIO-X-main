"""
CARDIO-X Framingham 10-Year CVD Risk Module (Placeholder).
"""

import streamlit as st


def render_framingham_page() -> None:
    """Renders the Framingham 10-Year CVD Risk Calculator placeholder."""
    st.markdown("### ⏱️ Framingham 10-Year Cardiovascular Disease Risk")
    st.markdown(
        """
        *Longitudinal primary prevention risk estimation predicting first incident hard CVD within 10 years*
        `TARGET_10Y_CVD = (First Incident Hard CVD within 3652.5 Days in Primary Prevention Cohort)`
        """
    )

    st.markdown("---")

    st.info(
        "🚧 **Module Under Active Development (Scheduled for Phase 3B)**\n\n"
        "The clinical primary-prevention calculator will evaluate baseline patient covariates:\n"
        "- Age and sex.\n"
        "- Total cholesterol, resting heart rate, and body mass index.\n"
        "- Blood pressure metrics: Systolic BP, Diastolic BP, and hypertension treatment status.\n"
        "- Smoking status and cigarettes per day.\n"
        "- Fasting glucose and diabetes diagnosis.\n\n"
        "Predictions will be computed using the pre-trained `models/framingham_risk.joblib` pipeline at the calibrated threshold $\\tau^* = 0.40$.",
        icon="🛠️"
    )

    st.markdown("#### Clinical Primary Prevention Target")
    c1, c2, c3 = st.columns(3)
    c1.metric("Study Cohort", "Framingham Heart Study", "Primary Prevention (N=3,983)")
    c2.metric("Target Sensitivity", "81.1%", "High Recall (tau=0.40)")
    c3.metric("Test ROC-AUC", "0.7648", "Standard BP Model")
