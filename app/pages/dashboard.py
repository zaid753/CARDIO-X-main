"""
CARDIO-X Dashboard View.
Renders executive overview, engine status badges, and academic workflow.
"""

from typing import Any, Dict
import streamlit as st

from app.utils.model_loader import load_model_metadata, verify_models_health
from app.components.status_cards import render_model_status_card


def render_dashboard_page() -> None:
    """Renders the comprehensive Phase 3A project dashboard."""
    st.markdown("### Executive System Overview")
    st.markdown(
        """
        **CARDIO-X** is an intelligent cardiovascular disease risk stratification and lipid analysis system.
        It implements a **dual-engine architecture** that decouples remote, non-invasive community screening triage
        from longitudinal clinical primary-prevention risk calculation.
        """
    )

    # Health & Verification Diagnostics
    health = verify_models_health()
    meta = load_model_metadata()
    artifacts_meta = meta.get("artifacts", {})

    st.markdown("#### Dual Machine Learning Engines Status")

    # Overview Metrics Row
    c1, c2, c3 = st.columns(3)
    c1.metric("Pipeline Integrity", "100% Operational" if health["overall_status"] == "healthy" else "Degraded", "Verified")
    c2.metric("Screening Features", "24 Non-Laboratory", "Core Battery")
    c3.metric("Clinical Risk Horizon", "10 Years", "Primary Prevention")

    st.markdown("<br>", unsafe_allow_html=True)

    # Engine 1: NHANES Core Screening Card
    if "nhanes_screening_core" in health["models"]:
        render_model_status_card(
            title="Engine 1A: Non-Laboratory Hypercholesterolemia Screening (Core 24)",
            model_key="nhanes_screening_core",
            model_data=health["models"]["nhanes_screening_core"],
            meta_data=artifacts_meta.get("nhanes_screening_core", {}),
        )

    # Engine 1B: NHANES Full Screening Card
    if "nhanes_screening_full" in health["models"]:
        with st.expander("Engine 1B: Candidate Pool Model (Full 58 Features)", expanded=False):
            render_model_status_card(
                title="Engine 1B: Non-Laboratory Screening (Full 58 Features)",
                model_key="nhanes_screening_full",
                model_data=health["models"]["nhanes_screening_full"],
                meta_data=artifacts_meta.get("nhanes_screening_full", {}),
            )

    # Engine 2: Framingham 10-Year CVD Risk Card
    if "framingham_risk" in health["models"]:
        render_model_status_card(
            title="Engine 2: 10-Year Incident CVD Primary Risk Model (Framingham)",
            model_key="framingham_risk",
            model_data=health["models"]["framingham_risk"],
            meta_data=artifacts_meta.get("framingham_risk", {}),
        )

    st.markdown("---")

    # Intended Clinical Workflow Section
    st.markdown("#### Intended End-to-End Clinical Screening Workflow")

    col_w1, col_w2, col_w3, col_w4 = st.columns(4)

    with col_w1:
        st.markdown(
            """
            <div style="border: 1px solid #E5E7EB; border-radius: 8px; padding: 14px; background: #F9FAFB; height: 100%;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 6px;">
                    1. Community Triage
                </div>
                <p style="font-size: 0.82rem; color: #4B5563; margin: 0;">
                    Patient enters 24 non-invasive biometrics (demographics, blood pressure, vitals, lifestyle habits).
                    Screening model identifies probable dyslipidemia without a blood draw.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_w2:
        st.markdown(
            """
            <div style="border: 1px solid #E5E7EB; border-radius: 8px; padding: 14px; background: #F9FAFB; height: 100%;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 6px;">
                    2. Lab Verification
                </div>
                <p style="font-size: 0.82rem; color: #4B5563; margin: 0;">
                    High-risk screening triage directs patient for confirmatory fasting lipid profile
                    (Total Cholesterol, HDL, LDL, Triglycerides).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_w3:
        st.markdown(
            """
            <div style="border: 1px solid #E5E7EB; border-radius: 8px; padding: 14px; background: #F9FAFB; height: 100%;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 6px;">
                    3. 10-Year CVD Risk
                </div>
                <p style="font-size: 0.82rem; color: #4B5563; margin: 0;">
                    Engine 2 evaluates baseline clinical measurements against the Framingham 10-year
                    landmark primary prevention model to estimate absolute CVD risk.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_w4:
        st.markdown(
            """
            <div style="border: 1px solid #E5E7EB; border-radius: 8px; padding: 14px; background: #F9FAFB; height: 100%;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 6px;">
                    4. Guidance & Report
                </div>
                <p style="font-size: 0.82rem; color: #4B5563; margin: 0;">
                    Automated generation of lifestyle recommendations (AHA guidelines) and clinical
                    summary reports for primary care consultation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Architectural Distinctions Callout
    with st.expander("Key Architectural Design Principles", expanded=True):
        st.markdown(
            """
            - **Strict Dataset Independence:** The NHANES screening model and Framingham risk model maintain
              independent preprocessing pipelines, target labels, and evaluation cohorts. Datasets are never merged.
            - **Zero Target Leakage:** Direct target-defining variables (e.g. cholesterol labs and statin medications)
              are strictly excluded from the non-invasive screening predictor matrix.
            - **Optimized Clinical Operating Thresholds:** Operating decision thresholds are frozen strictly from
              training-set out-of-fold cross-validation predictions targeting $\\ge 80\\%$ sensitivity for triage safety.
            """
        )

    # Clinical Governance Notice
    st.info(
        "**Governance & Academic Scope Notice:** CARDIO-X is an academic capstone demonstration. "
        "Predictions, performance metrics, and risk strata are derived from retrospective population cohorts "
        "(NHANES 2021-2023 and Framingham Heart Study) and must not be used as clinical diagnostic instruments.",
        icon="ℹ️"
    )
