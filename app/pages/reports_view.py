"""
CARDIO-X Reports and Clinical Audit Module (Placeholder).
"""

import streamlit as st


def render_reports_page() -> None:
    """Renders the Reports and Audit Module placeholder."""
    st.markdown("### 📑 Clinical Screening Reports and Documentation")
    st.markdown(
        """
        *Structured assessment reports and epidemiological evaluation documentation*
        """
    )

    st.markdown("---")

    st.info(
        "🚧 **Module Under Active Development (Scheduled for Phase 4)**\n\n"
        "This module will provide downloadable clinical summary reports:\n"
        "- Comprehensive PDF patient screening dossiers.\n"
        "- Model risk explanation breakdowns and feature importance profiles.\n"
        "- Provider communication summaries for physician consults.\n"
        "- Epidemiological benchmark comparison summaries from Phase 2B.\n\n"
        "PDF generation and export capabilities will be integrated in Phase 4.",
        icon="🛠️"
    )

    st.markdown("#### Available Evaluation Deliverables")
    st.markdown(
        """
        - [`reports/model_evaluation/model_evaluation_report.md`](file:///Users/zaid/Desktop/CARDIO-X-main/reports/model_evaluation/model_evaluation_report.md)
        - [`reports/model_evaluation/nhanes_core_comparison.csv`](file:///Users/zaid/Desktop/CARDIO-X-main/reports/model_evaluation/nhanes_core_comparison.csv)
        - [`reports/model_evaluation/framingham_comparison.csv`](file:///Users/zaid/Desktop/CARDIO-X-main/reports/model_evaluation/framingham_comparison.csv)
        """
    )
