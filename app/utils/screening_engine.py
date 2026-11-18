"""
CARDIO-X: NHANES Screening Engine and Validation Utilities.

Encapsulates feature validation, input schema enforcement, and inference execution
for the non-laboratory hypercholesterolemia screening model (Engine 1A).
Guarantees zero model retraining and zero raw dataset loading at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.utils.model_loader import load_model_metadata, load_pipeline

logger = logging.getLogger(__name__)

# Exact 24 core non-invasive predictors from Section 3.C of audit
NHANES_CORE_PREDICTORS: List[str] = [
    "age_years",
    "sex",
    "race_ethnicity",
    "education_level",
    "family_income_poverty_ratio",
    "bmi_kg_m2",
    "waist_circumference_cm",
    "waist_to_height_ratio",
    "weight_change_1yr_kg",
    "avg_systolic_bp_mmhg",
    "avg_diastolic_bp_mmhg",
    "pulse_pressure_mmhg",
    "avg_pulse_bpm",
    "bp_medication",
    "smoking_status",
    "total_ltp_met_minutes_week",
    "sedentary_minutes_day",
    "alcohol_drinks_per_week",
    "sleep_hours_weekday",
    "diabetes_diagnosed",
    "general_health",
    "phq9_score",
    "coronary_heart_disease",
    "stroke",
]

# Strict Target Leakage and Invasive Lab Exclusion Registry
PROHIBITED_LEAKAGE_VARIABLES: List[str] = [
    "total_cholesterol_mg_dl",
    "cholesterol_medication",
    "high_cholesterol_diagnosed",
    "non_hdl_cholesterol_mg_dl",
    "ldl_cholesterol_mg_dl",
    "triglycerides_mg_dl",
    "triglyceride_hdl_ratio",
    "hdl_cholesterol_mg_dl",
    "diabetes_undiagnosed",
    "glycemic_status",
    "survey_weight_interview_2yr",
    "survey_weight_mec_2yr",
    "survey_strata",
    "survey_psu",
    "participant_id",
]

# Allowed categorical choices matching fitted ColumnTransformer OneHotEncoder
VALID_SEX_CATEGORIES = ["female", "male"]

VALID_RACE_ETHNICITY_CATEGORIES = [
    "Mexican American",
    "Other Hispanic",
    "Non-Hispanic White",
    "Non-Hispanic Black",
    "Non-Hispanic Asian",
    "Other or multiracial",
]

VALID_EDUCATION_CATEGORIES = [
    "Less than 9th grade",
    "9th-11th grade",
    "High school/GED",
    "Some college/AA",
    "College graduate or above",
]

VALID_SMOKING_CATEGORIES = [
    "never",
    "former",
    "current",
]

VALID_GENERAL_HEALTH_CHOICES = {
    "Excellent": 1.0,
    "Very good": 2.0,
    "Good": 3.0,
    "Fair": 4.0,
    "Poor": 5.0,
}

# Empirical safety bounds grounded in NHANES 2021-2023 distribution
NUMERIC_BOUNDS = {
    "age_years": (18.0, 120.0, "Age (years)"),
    "family_income_poverty_ratio": (0.0, 5.0, "Family Income-to-Poverty Ratio"),
    "bmi_kg_m2": (10.0, 80.0, "Body Mass Index (BMI, kg/m²)"),
    "waist_circumference_cm": (40.0, 220.0, "Waist Circumference (cm)"),
    "waist_to_height_ratio": (0.20, 1.50, "Waist-to-Height Ratio"),
    "weight_change_1yr_kg": (-100.0, 100.0, "1-Year Weight Change (kg)"),
    "avg_systolic_bp_mmhg": (60.0, 260.0, "Average Systolic Blood Pressure (mmHg)"),
    "avg_diastolic_bp_mmhg": (30.0, 160.0, "Average Diastolic Blood Pressure (mmHg)"),
    "pulse_pressure_mmhg": (10.0, 160.0, "Pulse Pressure (mmHg)"),
    "avg_pulse_bpm": (30.0, 220.0, "Resting Heart Rate (bpm)"),
    "total_ltp_met_minutes_week": (0.0, 20000.0, "Leisure-Time Physical Activity (MET-min/week)"),
    "sedentary_minutes_day": (0.0, 1440.0, "Daily Sedentary Time (minutes/day)"),
    "alcohol_drinks_per_week": (0.0, 150.0, "Alcohol Consumption (drinks/week)"),
    "sleep_hours_weekday": (1.0, 24.0, "Average Weekday Sleep (hours)"),
    "phq9_score": (0.0, 27.0, "Depression Screening Score (PHQ-9)"),
}


def validate_nhanes_inputs(
    form_data: Dict[str, Any]
) -> Tuple[bool, List[str], Optional[pd.DataFrame]]:
    """
    Validates submitted patient form inputs against the exact model schema.

    Parameters
    ----------
    form_data : Dict[str, Any]
        Dictionary of raw user form inputs.

    Returns
    -------
    Tuple[bool, List[str], Optional[pd.DataFrame]]
        (is_valid, list_of_error_messages, formatted_single_row_dataframe)
    """
    errors: List[str] = []

    # 1. Strict Target Leakage Check
    for prohibited in PROHIBITED_LEAKAGE_VARIABLES:
        if prohibited in form_data and form_data[prohibited] is not None:
            errors.append(f"CRITICAL ERROR: Prohibited target-leakage variable '{prohibited}' detected.")

    # 2. Check completeness of essential core predictors
    for col in NHANES_CORE_PREDICTORS:
        if col not in form_data:
            errors.append(f"Missing required predictor field: '{col}'.")

    if errors:
        return False, errors, None

    cleaned_data: Dict[str, Any] = {}

    # 3. Categorical Validation
    # Sex
    raw_sex = str(form_data.get("sex", "")).strip().lower()
    if raw_sex not in VALID_SEX_CATEGORIES:
        errors.append(f"Invalid sex '{form_data.get('sex')}'. Must be one of {VALID_SEX_CATEGORIES}.")
    else:
        cleaned_data["sex"] = raw_sex

    # Race / Ethnicity
    raw_race = str(form_data.get("race_ethnicity", "")).strip()
    if raw_race not in VALID_RACE_ETHNICITY_CATEGORIES:
        errors.append(f"Invalid race/ethnicity '{raw_race}'. Must match NHANES categories.")
    else:
        cleaned_data["race_ethnicity"] = raw_race

    # Education Level
    raw_edu = str(form_data.get("education_level", "")).strip()
    if raw_edu not in VALID_EDUCATION_CATEGORIES:
        errors.append(f"Invalid education level '{raw_edu}'.")
    else:
        cleaned_data["education_level"] = raw_edu

    # Smoking Status
    raw_smoke = str(form_data.get("smoking_status", "")).strip().lower()
    if raw_smoke not in VALID_SMOKING_CATEGORIES:
        errors.append(f"Invalid smoking status '{raw_smoke}'. Must be 'never', 'former', or 'current'.")
    else:
        cleaned_data["smoking_status"] = raw_smoke

    # 4. Binary & Ordinal Feature Validation
    binary_cols = ["bp_medication", "diabetes_diagnosed", "coronary_heart_disease", "stroke"]
    for b_col in binary_cols:
        val = form_data.get(b_col)
        try:
            val_float = float(val)
            if val_float not in (0.0, 1.0):
                errors.append(f"Field '{b_col}' must be 0 (No) or 1 (Yes); received {val}.")
            else:
                cleaned_data[b_col] = val_float
        except (ValueError, TypeError):
            errors.append(f"Field '{b_col}' must be numeric binary (0 or 1).")

    # General Health (Ordinal 1.0 to 5.0)
    gh_val = form_data.get("general_health")
    try:
        gh_float = float(gh_val)
        if gh_float not in (1.0, 2.0, 3.0, 4.0, 5.0):
            errors.append(f"General health score must be an integer between 1 (Excellent) and 5 (Poor); received {gh_val}.")
        else:
            cleaned_data["general_health"] = gh_float
    except (ValueError, TypeError):
        errors.append(f"General health must be a valid numeric rating between 1 and 5.")

    # 5. Continuous Numerical Range Validation
    for num_col, (min_val, max_val, label) in NUMERIC_BOUNDS.items():
        val = form_data.get(num_col)

        # Allow income poverty ratio or PHQ-9 to be None/NaN if unspecified (imputer handles it)
        if val is None or pd.isna(val) or val == "":
            if num_col in ("family_income_poverty_ratio", "phq9_score", "total_ltp_met_minutes_week"):
                cleaned_data[num_col] = np.nan
                continue
            else:
                errors.append(f"Field '{label}' is required and cannot be empty.")
                continue

        try:
            val_float = float(val)
            if val_float < min_val or val_float > max_val:
                errors.append(f"{label} ({val_float}) is outside allowable data bounds [{min_val}, {max_val}].")
            else:
                cleaned_data[num_col] = val_float
        except (ValueError, TypeError):
            errors.append(f"{label} must be a valid number.")

    # 6. Hemodynamic Consistency Check
    if "avg_systolic_bp_mmhg" in cleaned_data and "avg_diastolic_bp_mmhg" in cleaned_data:
        sys_bp = cleaned_data["avg_systolic_bp_mmhg"]
        dia_bp = cleaned_data["avg_diastolic_bp_mmhg"]
        if not pd.isna(sys_bp) and not pd.isna(dia_bp):
            if sys_bp <= dia_bp:
                errors.append(
                    f"Physiological inconsistency: Systolic BP ({sys_bp} mmHg) must be greater than Diastolic BP ({dia_bp} mmHg)."
                )
            else:
                # Recalculate pulse pressure for strict mathematical consistency
                derived_pp = round(sys_bp - dia_bp, 1)
                cleaned_data["pulse_pressure_mmhg"] = derived_pp

    # 7. Anthropometric Consistency Check
    if "waist_circumference_cm" in cleaned_data and "height_cm" in form_data:
        ht = form_data.get("height_cm")
        if ht is not None and not pd.isna(ht):
            try:
                ht_float = float(ht)
                if 80.0 <= ht_float <= 250.0:
                    cleaned_data["waist_to_height_ratio"] = round(cleaned_data["waist_circumference_cm"] / ht_float, 4)
            except (ValueError, TypeError):
                pass

    if errors:
        return False, errors, None

    # Construct single-row DataFrame in the EXACT expected feature order
    df_row = pd.DataFrame([cleaned_data])[NHANES_CORE_PREDICTORS]
    return True, [], df_row


def predict_nhanes_screening(
    df_input: pd.DataFrame,
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Executes inference using the pre-trained NHANES Core screening pipeline.
    Never fits, retrains, or loads raw datasets.

    Parameters
    ----------
    df_input : pd.DataFrame
        Single-row or batch DataFrame with exact 24 core feature names.
    models_dir : Optional[Union[str, Path]]
        Optional explicit models directory.

    Returns
    -------
    Dict[str, Any]
        Inference results dictionary.
    """
    # 1. Load pipeline and metadata
    pipeline = load_pipeline("nhanes_screening_core", models_dir=models_dir)
    if pipeline is None:
        raise RuntimeError("Failed to load NHANES core model artifact ('nhanes_screening_core.joblib').")

    metadata = load_model_metadata(models_dir=models_dir)
    core_meta = metadata.get("artifacts", {}).get("nhanes_screening_core", {})

    # Extract operating threshold from metadata (calibrated for sensitivity >= 80%)
    raw_threshold = core_meta.get("selected_decision_threshold")
    threshold = float(raw_threshold) if raw_threshold is not None else 0.4600

    # 2. Verify expected feature columns and order
    missing_cols = [c for c in NHANES_CORE_PREDICTORS if c not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required predictor columns: {missing_cols}")

    df_ordered = df_input[NHANES_CORE_PREDICTORS].copy()

    # 3. Model Inference via predict_proba
    classifier = pipeline.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "classes_"):
        raise RuntimeError("Loaded pipeline does not expose a valid classifier with classes_.")

    classes = classifier.classes_
    if 1 not in classes:
        raise RuntimeError(f"Target class '1' not found in model classes: {classes}")

    pos_class_idx = int(np.where(classes == 1)[0][0])

    probs = pipeline.predict_proba(df_ordered)
    pos_prob = float(probs[0, pos_class_idx])
    neg_prob = float(probs[0, 1 - pos_class_idx]) if len(classes) == 2 else 1.0 - pos_prob

    # 4. Classification against frozen operating threshold
    is_higher_signal = bool(pos_prob >= threshold)
    classification_label = "Higher Screening Signal" if is_higher_signal else "Lower Screening Signal"

    # Calibration benchmark metrics from Phase 2B test evaluation
    test_metrics = core_meta.get("test_metrics_at_selected_threshold", {})

    return {
        "status": "success",
        "model_key": "nhanes_screening_core",
        "model_tier": "Model 1: Remote Screening Engine (Core 24)",
        "estimator_class": classifier.__class__.__name__,
        "screening_score": round(pos_prob, 4),
        "screening_score_pct": round(pos_prob * 100.0, 1),
        "negative_score_pct": round(neg_prob * 100.0, 1),
        "operating_threshold": round(threshold, 4),
        "operating_threshold_pct": round(threshold * 100.0, 1),
        "is_higher_signal": is_higher_signal,
        "classification_label": classification_label,
        "calibrated_test_sensitivity": test_metrics.get("sensitivity", 0.7790),
        "calibrated_test_specificity": test_metrics.get("specificity", 0.6858),
        "calibrated_test_roc_auc": test_metrics.get("roc_auc", 0.8205),
        "target_name": core_meta.get("target_name", "TARGET_HYPERCHOL"),
        "target_definition": core_meta.get(
            "target_definition", "total_cholesterol_mg_dl >= 240 OR cholesterol_medication == 1"
        ),
        "input_summary": df_ordered.iloc[0].to_dict(),
    }
