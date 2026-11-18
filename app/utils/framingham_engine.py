"""
CARDIO-X: Framingham 10-Year CVD Risk Engine and Validation Utilities.

Encapsulates feature validation, input schema enforcement, and inference execution
for the longitudinal 10-year incident cardiovascular disease model (Model 2).
Guarantees zero model retraining and zero raw dataset loading at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.utils.model_loader import load_model_metadata, load_pipeline

logger = logging.getLogger(__name__)

# Exact 14 baseline predictors required by models/framingham_risk.joblib
FRAMINGHAM_CORE_PREDICTORS: List[str] = [
    "AGE",
    "SEX",
    "CURSMOKE",
    "CIGPDAY",
    "educ",
    "BPMEDS",
    "PREVHYP",
    "HEARTRTE",
    "TOTCHOL",
    "BMI",
    "DIABETES",
    "GLUCOSE",
    "SYSBP",
    "DIABP",
]

# Prohibited Target Leakage, Longitudinal Surveillance, and Future Outcome Variables
PROHIBITED_LEAKAGE_VARIABLES: List[str] = [
    "TARGET_10Y_CVD",
    "CVD",
    "ANYCHD",
    "CORONARY",
    "STROKE",
    "HOSPMI",
    "MI_FCHD",
    "TIMECVD",
    "TIMEDTH",
    "TIMEHYP",
    "TIMECHD",
    "TIMESTRK",
    "DEATH",
    "RANDID",
    "PERIOD",
    "HDLC",
    "LDLC",
    "TRIG",
    "triglycerides",
    "PULSE_PRESSURE",
    "MAP",
]

# Allowed categorical & binary options matching the fitted ColumnTransformer
VALID_SEX_CHOICES: Dict[str, float] = {
    "Female": 0.0,
    "Male": 1.0,
}

VALID_EDUC_CHOICES: Dict[str, float] = {
    "0–11 Years / Some High School": 1.0,
    "High School Diploma / GED": 2.0,
    "Some College / Vocational School": 3.0,
    "College Graduate or Higher": 4.0,
}

# Empirical biological safety bounds grounded in Framingham Heart Study distribution
NUMERIC_BOUNDS: Dict[str, Tuple[float, float, str]] = {
    "AGE": (30.0, 85.0, "Age (years)"),
    "CIGPDAY": (0.0, 100.0, "Cigarettes per Day"),
    "SYSBP": (70.0, 260.0, "Systolic Blood Pressure (mmHg)"),
    "DIABP": (40.0, 150.0, "Diastolic Blood Pressure (mmHg)"),
    "HEARTRTE": (35.0, 180.0, "Resting Heart Rate (bpm)"),
    "TOTCHOL": (100.0, 600.0, "Total Cholesterol (mg/dL)"),
    "BMI": (12.0, 70.0, "Body Mass Index (BMI, kg/m²)"),
    "GLUCOSE": (40.0, 500.0, "Blood Glucose (mg/dL)"),
}


def validate_framingham_inputs(
    form_data: Dict[str, Any]
) -> Tuple[bool, List[str], Optional[pd.DataFrame]]:
    """
    Validates submitted patient form inputs against the exact Framingham model schema.

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
    for col in FRAMINGHAM_CORE_PREDICTORS:
        if col not in form_data:
            errors.append(f"Missing required predictor field: '{col}'.")

    if errors:
        return False, errors, None

    cleaned_data: Dict[str, Any] = {}

    # 3. Categorical / Demographic Validation
    # Sex: Female=0.0, Male=1.0
    sex_val = form_data.get("SEX")
    if sex_val is None or pd.isna(sex_val) or sex_val == "":
        errors.append("Field 'SEX' is required. Please explicitly select Female or Male.")
    else:
        try:
            sex_float = float(sex_val)
            if sex_float not in (0.0, 1.0):
                errors.append(f"Invalid SEX value '{sex_val}'. Must be 0 (Female) or 1 (Male).")
            else:
                cleaned_data["SEX"] = sex_float
        except (ValueError, TypeError):
            errors.append("Field 'SEX' must be a valid binary indicator (0 for Female, 1 for Male).")

    # Education level (educ): 1.0 to 4.0
    educ_val = form_data.get("educ")
    if educ_val is None or pd.isna(educ_val) or educ_val == "":
        # Allowed as missing; training pipeline imputes via most_frequent (mode = 1.0 / 2.0)
        cleaned_data["educ"] = np.nan
    else:
        try:
            educ_float = float(educ_val)
            if educ_float not in (1.0, 2.0, 3.0, 4.0):
                errors.append(f"Field 'educ' must be between 1 and 4; received {educ_val}.")
            else:
                cleaned_data["educ"] = educ_float
        except (ValueError, TypeError):
            errors.append("Field 'educ' must be an integer between 1 and 4.")

    # 4. Binary Feature Validation (BPMEDS, PREVHYP, DIABETES, CURSMOKE)
    binary_cols = ["BPMEDS", "PREVHYP", "DIABETES", "CURSMOKE"]
    for b_col in binary_cols:
        val = form_data.get(b_col)
        if val is None or pd.isna(val) or val == "":
            errors.append(f"Field '{b_col}' is required. Please explicitly select Yes or No.")
            continue
        try:
            val_float = float(val)
            if val_float not in (0.0, 1.0):
                errors.append(f"Field '{b_col}' must be 0 (No) or 1 (Yes); received {val}.")
            else:
                cleaned_data[b_col] = val_float
        except (ValueError, TypeError):
            errors.append(f"Field '{b_col}' must be numeric binary (0 or 1).")

    # 5. Continuous Numerical Range Validation
    for num_col, (min_val, max_val, label) in NUMERIC_BOUNDS.items():
        val = form_data.get(num_col)

        # Allow BMI, GLUCOSE, HEARTRTE to be None/NaN if unmeasured (imputer handles via median)
        if val is None or pd.isna(val) or val == "":
            if num_col in ("BMI", "GLUCOSE", "HEARTRTE"):
                cleaned_data[num_col] = np.nan
                continue
            elif num_col == "CIGPDAY" and cleaned_data.get("CURSMOKE") == 0.0:
                cleaned_data["CIGPDAY"] = 0.0
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

    # 6. Smoking Behavioral Consistency Check
    if "CURSMOKE" in cleaned_data and "CIGPDAY" in cleaned_data:
        cur_smoke = cleaned_data["CURSMOKE"]
        cigpday = cleaned_data["CIGPDAY"]
        if cur_smoke == 0.0 and cigpday > 0.0:
            errors.append("Consistency error: Non-smokers (CURSMOKE=0) must report 0 cigarettes per day.")
        elif cur_smoke == 1.0 and cigpday < 1.0:
            errors.append("Consistency error: Current smokers (CURSMOKE=1) must report at least 1 cigarette per day.")

    # 7. Hemodynamic Consistency Check
    if "SYSBP" in cleaned_data and "DIABP" in cleaned_data:
        sys_bp = cleaned_data["SYSBP"]
        dia_bp = cleaned_data["DIABP"]
        if not pd.isna(sys_bp) and not pd.isna(dia_bp):
            if sys_bp <= dia_bp:
                errors.append(
                    f"Physiological inconsistency: Systolic BP ({sys_bp} mmHg) must be greater than Diastolic BP ({dia_bp} mmHg)."
                )
            elif (sys_bp - dia_bp) < 10.0:
                errors.append(
                    f"Physiological inconsistency: Pulse pressure ({round(sys_bp - dia_bp, 1)} mmHg) is below biological threshold (minimum 10 mmHg)."
                )

    if errors:
        return False, errors, None

    # Construct single-row DataFrame in the EXACT expected feature order
    df_row = pd.DataFrame([cleaned_data])[FRAMINGHAM_CORE_PREDICTORS]
    return True, [], df_row


def predict_framingham_risk(
    df_input: pd.DataFrame,
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Executes inference using the pre-trained Framingham 10-year CVD risk pipeline.
    Never fits, retrains, or loads raw datasets.

    Parameters
    ----------
    df_input : pd.DataFrame
        Single-row or batch DataFrame with exact 14 core feature names.
    models_dir : Optional[Union[str, Path]]
        Optional explicit models directory.

    Returns
    -------
    Dict[str, Any]
        Inference results dictionary.
    """
    # 1. Load pipeline and metadata
    pipeline = load_pipeline("framingham_risk", models_dir=models_dir)
    if pipeline is None:
        raise RuntimeError("Failed to load Framingham risk model artifact ('framingham_risk.joblib').")

    metadata = load_model_metadata(models_dir=models_dir)
    fram_meta = metadata.get("artifacts", {}).get("framingham_risk", {})

    # Extract operating threshold from metadata (calibrated for sensitivity >= 80%)
    raw_threshold = fram_meta.get("selected_decision_threshold")
    threshold = float(raw_threshold) if raw_threshold is not None else 0.4000

    # 2. Verify expected feature columns and order
    missing_cols = [c for c in FRAMINGHAM_CORE_PREDICTORS if c not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required predictor columns: {missing_cols}")

    df_ordered = df_input[FRAMINGHAM_CORE_PREDICTORS].copy()

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
    classification_label = (
        "Higher Model Screening Signal"
        if is_higher_signal
        else "Lower Model Screening Signal"
    )

    # Calibration benchmark metrics from Phase 2B test evaluation
    test_metrics = fram_meta.get("test_metrics_at_selected_threshold", {})

    # Compute derived hemodynamic markers for clinical context
    sys_val = float(df_ordered["SYSBP"].iloc[0])
    dia_val = float(df_ordered["DIABP"].iloc[0])
    pp_val = round(sys_val - dia_val, 1)
    map_val = round(dia_val + (pp_val / 3.0), 1)

    return {
        "status": "success",
        "model_key": "framingham_risk",
        "model_tier": "Model 2: Longitudinal Risk Engine (Standard BP 14)",
        "estimator_class": classifier.__class__.__name__,
        "screening_score": round(pos_prob, 4),
        "screening_score_pct": round(pos_prob * 100.0, 1),
        "risk_score": round(pos_prob, 4),
        "risk_score_pct": round(pos_prob * 100.0, 1),
        "negative_score_pct": round(neg_prob * 100.0, 1),
        "operating_threshold": round(threshold, 4),
        "operating_threshold_pct": round(threshold * 100.0, 1),
        "is_higher_signal": is_higher_signal,
        "classification_label": classification_label,
        "calibrated_test_sensitivity": test_metrics.get("sensitivity", 0.8108),
        "calibrated_test_specificity": test_metrics.get("specificity", 0.5685),
        "calibrated_test_roc_auc": test_metrics.get("roc_auc", 0.7648),
        "calibrated_test_accuracy": test_metrics.get("accuracy", 0.5910),
        "calibrated_test_f1": test_metrics.get("f1_score", 0.2691),
        "target_name": fram_meta.get("target_name", "TARGET_10Y_CVD"),
        "target_definition": fram_meta.get(
            "target_definition",
            "First incident hard CVD event within 3652.5 days (10 years) in primary prevention cohort",
        ),
        "derived_hemodynamics": {
            "pulse_pressure_mmhg": pp_val,
            "mean_arterial_pressure_mmhg": map_val,
        },
        "input_summary": df_ordered.iloc[0].to_dict(),
    }
