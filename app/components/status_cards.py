"""
CARDIO-X Status Cards and Diagnostic Displays.
"""

from typing import Any, Dict
import streamlit as st


def render_model_status_card(
    title: str,
    model_key: str,
    model_data: Dict[str, Any],
    meta_data: Dict[str, Any],
) -> None:
    """
    Renders an individual model health card reading strictly from actual verified artifacts.
    """
    is_operational = model_data.get("status") == "operational"
    status_color = "#10B981" if is_operational else "#F59E0B"
    status_label = "OPERATIONAL" if is_operational else model_data.get("status", "UNKNOWN").upper()

    # Extract actual verified metrics from metadata
    test_opt = meta_data.get("test_metrics_at_selected_threshold", {})
    test_roc = test_opt.get("roc_auc", "N/A")
    test_sens = test_opt.get("sensitivity", "N/A")
    test_spec = test_opt.get("specificity", "N/A")
    test_brier = test_opt.get("brier_score", "N/A")

    if isinstance(test_sens, float):
        sens_str = f"{test_sens * 100:.1f}%"
    else:
        sens_str = str(test_sens)

    if isinstance(test_spec, float):
        spec_str = f"{test_spec * 100:.1f}%"
    else:
        spec_str = str(test_spec)

    st.markdown(
        f"""
        <div style="border: 1px solid #E5E7EB; border-radius: 8px; padding: 16px; margin-bottom: 12px;
                    background-color: #FFFFFF; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h4 style="margin: 0; color: #111827; font-size: 1.05rem;">{title}</h4>
                <span style="background-color: {status_color}20; color: {status_color}; font-weight: 700;
                             padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; letter-spacing: 0.5px;">
                    {status_label}
                </span>
            </div>
            <p style="margin: 0 0 10px 0; font-size: 0.82rem; color: #4B5563;">
                <strong>Architecture:</strong> {model_data.get("estimator_class", "Not loaded")} &bull;
                <strong>Predictors:</strong> {model_data.get("feature_count", 0)} non-invasive features &bull;
                <strong>Operating Threshold:</strong> &tau; = {model_data.get('operating_threshold', 0.50):.2f}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test ROC-AUC", f"{test_roc:.4f}" if isinstance(test_roc, float) else test_roc)
    c2.metric("Target Sensitivity", sens_str)
    c3.metric("Test Specificity", spec_str)
    c4.metric("Brier Loss", f"{test_brier:.4f}" if isinstance(test_brier, float) else test_brier)
