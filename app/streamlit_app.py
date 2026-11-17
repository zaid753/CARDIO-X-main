"""
CARDIO-X: Intelligent Cardiovascular Risk Screening and Lipid Profile Analysis System.
Main Streamlit Application Entrypoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct module discovery
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st

from app.components.header import render_header
from app.components.sidebar import render_sidebar
from app.pages import (
    render_dashboard_page,
    render_nhanes_page,
    render_framingham_page,
    render_lipid_page,
    render_lifestyle_page,
    render_reports_page,
)


def configure_page() -> None:
    """Sets standard Streamlit page configuration and styling."""
    st.set_page_config(
        page_title="CARDIO-X | Cardiovascular Screening & Risk System",
        page_icon="🫀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for clean, professional typography and layout
    st.markdown(
        """
        <style>
            /* Main container padding */
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
                max-width: 1200px;
            }
            /* Card and container styling */
            div[data-testid="stMetricValue"] {
                font-size: 1.5rem;
                font-weight: 700;
                color: #1E3A8A;
            }
            div[data-testid="stMetricLabel"] {
                font-size: 0.85rem;
                font-weight: 500;
                color: #4B5563;
            }
            /* Clean button aesthetics */
            .stButton>button {
                border-radius: 6px;
                font-weight: 600;
            }
            /* Sidebar header tweaks */
            section[data-testid="stSidebar"] {
                background-color: #F8FAFC;
                border-right: 1px solid #E2E8F0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Main application lifecycle and page routing controller."""
    configure_page()
    render_header()
    selected_page = render_sidebar()

    # Page Dispatcher
    if "Dashboard" in selected_page:
        render_dashboard_page()
    elif "NHANES" in selected_page:
        render_nhanes_page()
    elif "Framingham" in selected_page:
        render_framingham_page()
    elif "Lipid" in selected_page:
        render_lipid_page()
    elif "Lifestyle" in selected_page:
        render_lifestyle_page()
    elif "Reports" in selected_page:
        render_reports_page()
    else:
        render_dashboard_page()


if __name__ == "__main__":
    main()
