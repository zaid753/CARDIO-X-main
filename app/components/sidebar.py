"""
CARDIO-X Sidebar Navigation Component.
"""

from typing import Tuple
import streamlit as st
from app.utils.model_loader import verify_models_health

PAGES = [
    "📊 Dashboard",
    "🧬 NHANES Cholesterol Screening",
    "⏱️ Framingham 10-Year CVD Risk",
    "🩸 Lipid Profile Analysis",
    "🥗 Lifestyle Recommendations",
    "📑 Reports & Audit",
]


def render_sidebar() -> str:
    """
    Renders the persistent sidebar menu, health indicators, and legal/clinical disclaimer.

    Returns
    -------
    str
        Selected page title.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding-bottom: 15px;">
                <h2 style="color: #1E3A8A; margin-bottom: 0;">CARDIO-X</h2>
                <span style="font-size: 0.8rem; color: #6B7280; font-weight: 500;">
                    Cardiovascular Screening System
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Navigation")
        selected_page = st.radio(
            "Select Module",
            options=PAGES,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Runtime Engine Status Mini-Badge
        st.markdown("#### System Health")
        health = verify_models_health()
        if health["overall_status"] == "healthy":
            st.success("All ML Pipelines Operational", icon="✅")
        elif health["overall_status"] == "degraded":
            st.warning("Some Models Degraded/Missing", icon="⚠️")
        else:
            st.error("Engine Offline", icon="❌")

        # Independent Modules Indicator
        with st.expander("Engine Architectures", expanded=False):
            st.markdown(
                """
                - **Model 1 (NHANES):** Non-laboratory dyslipidemia screening triage (HistGradientBoosting / Logistic Regression).
                - **Model 2 (Framingham):** 10-year incident hard CVD primary risk prediction (L2 Logistic Regression).
                """
            )

        st.markdown("---")

        # Mandatory Clinical Disclaimer
        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #6B7280; line-height: 1.35; padding: 6px;
                        background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px;">
                <strong>Academic Prototype Notice:</strong><br>
                CARDIO-X is an academic research screening prototype developed for a final-year BTech project.
                It is <em>not</em> an FDA-cleared diagnostic device and must not replace professional clinical judgment.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("CARDIO-X &copy; 2026 &bull; Final Year Capstone")

    return selected_page
