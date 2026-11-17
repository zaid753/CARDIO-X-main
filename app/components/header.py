"""
CARDIO-X Application Header Component.
"""

import streamlit as st


def render_header() -> None:
    """Renders the top application brand header and status indicators."""
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown(
            """
            <div style="padding-top: 5px; padding-bottom: 10px;">
                <h1 style="margin: 0; color: #1E3A8A; font-size: 2.3rem; font-weight: 700; letter-spacing: -0.5px;">
                    🫀 CARDIO-X
                </h1>
                <p style="margin: 4px 0 0 0; color: #4B5563; font-size: 1.05rem; font-weight: 500;">
                    Intelligent Cardiovascular Risk Screening and Lipid Profile Analysis System
                </p>
                <p style="margin: 2px 0 0 0; color: #6B7280; font-size: 0.85rem;">
                    BTech Final-Year Capstone Project &bull; Dual-Engine Machine Learning Architecture
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 15px;">
                <span style="background-color: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE;
                             padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">
                    v1.0.0-phase3a
                </span>
                <p style="margin: 5px 0 0 0; font-size: 0.75rem; color: #9CA3AF;">Academic Prototype</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px; border: 0; border-top: 1px solid #E5E7EB;'>", unsafe_allow_html=True)
