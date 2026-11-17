"""CARDIO-X Application Page Views."""
from app.pages.dashboard import render_dashboard_page
from app.pages.nhanes_screening import render_nhanes_page
from app.pages.framingham_risk import render_framingham_page
from app.pages.lipid_analysis import render_lipid_page
from app.pages.lifestyle import render_lifestyle_page
from app.pages.reports_view import render_reports_page

__all__ = [
    "render_dashboard_page",
    "render_nhanes_page",
    "render_framingham_page",
    "render_lipid_page",
    "render_lifestyle_page",
    "render_reports_page",
]
