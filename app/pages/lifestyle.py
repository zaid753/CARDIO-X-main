"""
CARDIO-X Evidence-Based Lifestyle Recommendations Module (Placeholder).
"""

import streamlit as st


def render_lifestyle_page() -> None:
    """Renders the Evidence-Based Lifestyle Recommendations placeholder."""
    st.markdown("### 🥗 Evidence-Based Lifestyle Recommendations")
    st.markdown(
        """
        *Personalized dietary, physical activity, and risk modification guidance aligned with ACC/AHA guidelines*
        """
    )

    st.markdown("---")

    st.info(
        "🚧 **Module Under Active Development (Scheduled for Phase 4)**\n\n"
        "This component will generate evidence-based behavioral intervention pathways based on patient risk tier:\n"
        "- **Dietary Interventions:** Mediterranean and DASH dietary patterns, saturated fat restriction, soluble fiber augmentation.\n"
        "- **Physical Activity:** Aerobic exercise prescriptions targeting $\\ge 150$ moderate-intensity MET-minutes/week.\n"
        "- **Blood Pressure & Weight Management:** Sodium reduction and caloric deficit strategies.\n"
        "- **Smoking Cessation & Sleep Hygiene:** Behavioral counseling algorithms and sleep duration targets.\n\n"
        "All recommendations are parameterized by the patient's specific metabolic profile without automated pharmacotherapy prescription.",
        icon="🛠️"
    )

    st.markdown("#### Clinical Guidelines Aligned")
    st.markdown(
        """
        - **AHA/ACC 2019 Primary Prevention Guidelines** on Cardiovascular Disease.
        - **2018 AHA/ACC/AACVPR/AAPA/ABC/ACPM/ADA/AGS/APhA/ASPC/NLA/PCNA** Guideline on the Management of Blood Cholesterol.
        - **WHO Guidelines** on Physical Activity and Sedentary Behaviour.
        """
    )
