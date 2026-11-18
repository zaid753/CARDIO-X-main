"""
Automated unit and integration test suite for the CARDIO-X NHANES Screening Module (Phase 3B).
Verifies:
1. Feature schema exact match with saved NHANES core model.
2. All 24 core predictors represented exactly once.
3. Zero target leakage or prohibited laboratory features.
4. Input DataFrame column ordering.
5. Numeric and categorical validation checks.
6. Missing input handling (required vs. imputable optional fields).
7. Rejection of unknown categories.
8. Positive-class probability extraction using model.classes_.
9. Correct threshold-based classification using metadata.
10. Zero retraining or model modification during inference.
11. Graceful error handling on missing/corrupted artifacts.
12. Safe prediction error handling.
13. Clean imports of all application pages and components.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.utils.model_loader import load_pipeline, load_model_metadata, get_models_dir
from app.utils.screening_engine import (
    NHANES_CORE_PREDICTORS,
    PROHIBITED_LEAKAGE_VARIABLES,
    VALID_SEX_CATEGORIES,
    VALID_RACE_ETHNICITY_CATEGORIES,
    VALID_EDUCATION_CATEGORIES,
    VALID_SMOKING_CATEGORIES,
    VALID_GENERAL_HEALTH_CHOICES,
    NUMERIC_BOUNDS,
    validate_nhanes_inputs,
    predict_nhanes_screening,
)
from app.pages.nhanes_screening import render_nhanes_page
from app.pages.dashboard import render_dashboard_page
from app.pages.framingham_risk import render_framingham_page
from app.pages.lipid_analysis import render_lipid_page
from app.pages.lifestyle import render_lifestyle_page
from app.pages.reports_view import render_reports_page


class TestNHANESInteractiveScreening(unittest.TestCase):
    """Test suite for Phase 3B NHANES Screening Module."""

    def setUp(self):
        """Prepare valid standard test payload grounded in NHANES medians."""
        self.valid_payload: Dict[str, Any] = {
            "age_years": 54.0,
            "sex": "female",
            "race_ethnicity": "Non-Hispanic White",
            "education_level": "College graduate or above",
            "family_income_poverty_ratio": 3.0,
            "bmi_kg_m2": 28.2,
            "waist_circumference_cm": 95.0,
            "height_cm": 165.0,
            "waist_to_height_ratio": 0.5758,
            "weight_change_1yr_kg": 0.0,
            "avg_systolic_bp_mmhg": 124.0,
            "avg_diastolic_bp_mmhg": 78.0,
            "pulse_pressure_mmhg": 46.0,
            "avg_pulse_bpm": 70.0,
            "bp_medication": 0.0,
            "smoking_status": "never",
            "total_ltp_met_minutes_week": 480.0,
            "sedentary_minutes_day": 360.0,
            "alcohol_drinks_per_week": 1.0,
            "sleep_hours_weekday": 7.5,
            "diabetes_diagnosed": 0.0,
            "general_health": 2.0,
            "phq9_score": 1.0,
            "coronary_heart_disease": 0.0,
            "stroke": 0.0,
        }

    def test_all_24_core_predictors_represented_exactly_once(self):
        """Verify that exactly 24 unique core predictors are specified."""
        self.assertEqual(len(NHANES_CORE_PREDICTORS), 24)
        self.assertEqual(len(set(NHANES_CORE_PREDICTORS)), 24)

    def test_feature_schema_matches_saved_core_model(self):
        """Verify that the 24 predictors match the saved pipeline preprocessor expectations."""
        pipeline = load_pipeline("nhanes_screening_core")
        self.assertIsNotNone(pipeline)
        prep = pipeline.named_steps["preprocessor"]

        expected_pipeline_features = []
        for name, transformer, cols in prep.transformers_:
            expected_pipeline_features.extend(cols)

        self.assertEqual(sorted(NHANES_CORE_PREDICTORS), sorted(expected_pipeline_features))

    def test_zero_prohibited_leakage_features_in_schema(self):
        """Verify no target definers, lipid labs, or survey weights are in the predictor schema."""
        for prohibited in PROHIBITED_LEAKAGE_VARIABLES:
            self.assertNotIn(prohibited, NHANES_CORE_PREDICTORS)
            self.assertNotIn("cholesterol", prohibited.lower()) if prohibited in NHANES_CORE_PREDICTORS else None

        # Verify validation rejects payloads with prohibited variables
        leakage_payload = self.valid_payload.copy()
        leakage_payload["total_cholesterol_mg_dl"] = 220.0
        is_valid, errors, _ = validate_nhanes_inputs(leakage_payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("leakage" in err.lower() for err in errors))

    def test_input_data_constructed_in_exact_column_order(self):
        """Verify validated output DataFrame matches the exact column list and order."""
        is_valid, errors, df_validated = validate_nhanes_inputs(self.valid_payload)
        self.assertTrue(is_valid, f"Validation failed unexpectedly: {errors}")
        self.assertIsNotNone(df_validated)
        self.assertEqual(list(df_validated.columns), NHANES_CORE_PREDICTORS)
        self.assertEqual(len(df_validated), 1)

    def test_numeric_validation_and_hemodynamic_checks(self):
        """Verify physiological and numerical boundaries are strictly enforced."""
        # 1. Systolic BP <= Diastolic BP must fail
        bad_bp_payload = self.valid_payload.copy()
        bad_bp_payload["avg_systolic_bp_mmhg"] = 75.0
        bad_bp_payload["avg_diastolic_bp_mmhg"] = 80.0
        is_valid, errors, _ = validate_nhanes_inputs(bad_bp_payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("systolic bp" in err.lower() for err in errors))

        # 2. Out-of-bounds Age must fail
        bad_age_payload = self.valid_payload.copy()
        bad_age_payload["age_years"] = 15.0  # Min is 18
        is_valid, errors, _ = validate_nhanes_inputs(bad_age_payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("age" in err.lower() for err in errors))

        # 3. Negative pulse must fail
        bad_pulse_payload = self.valid_payload.copy()
        bad_pulse_payload["avg_pulse_bpm"] = -10.0
        is_valid, errors, _ = validate_nhanes_inputs(bad_pulse_payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("resting heart rate" in err.lower() for err in errors))

    def test_categorical_validation_and_rejection_of_unknown_categories(self):
        """Verify invalid or un-modeled categorical strings are rejected."""
        # Invalid sex
        bad_sex_payload = self.valid_payload.copy()
        bad_sex_payload["sex"] = "unknown_alien"
        is_valid, errors, _ = validate_nhanes_inputs(bad_sex_payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("sex" in err.lower() for err in errors))

        # Invalid race
        bad_race_payload = self.valid_payload.copy()
        bad_race_payload["race_ethnicity"] = "Martian Demographics"
        is_valid, errors, _ = validate_nhanes_inputs(bad_race_payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("race/ethnicity" in err.lower() for err in errors))

        # Invalid smoking status
        bad_smoke_payload = self.valid_payload.copy()
        bad_smoke_payload["smoking_status"] = "vaping_only"
        is_valid, errors, _ = validate_nhanes_inputs(bad_smoke_payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("smoking" in err.lower() for err in errors))

    def test_missing_required_inputs_handling(self):
        """Verify missing required fields fail validation with clear messages."""
        missing_payload = self.valid_payload.copy()
        del missing_payload["age_years"]
        del missing_payload["sex"]
        is_valid, errors, _ = validate_nhanes_inputs(missing_payload)
        self.assertFalse(is_valid)
        self.assertGreaterEqual(len(errors), 2)
        self.assertTrue(any("age_years" in err for err in errors))
        self.assertTrue(any("sex" in err for err in errors))

    def test_optional_imputable_inputs_allow_nan(self):
        """Verify optional fields like PIR or PHQ-9 allow NaN for imputer handling."""
        nan_payload = self.valid_payload.copy()
        nan_payload["family_income_poverty_ratio"] = np.nan
        nan_payload["phq9_score"] = np.nan
        is_valid, errors, df_validated = validate_nhanes_inputs(nan_payload)
        self.assertTrue(is_valid, f"Validation failed with NaNs in optional fields: {errors}")
        self.assertTrue(pd.isna(df_validated["family_income_poverty_ratio"].iloc[0]))
        self.assertTrue(pd.isna(df_validated["phq9_score"].iloc[0]))

        # Confirm pipeline predicts successfully on this row using fitted median imputation
        res = predict_nhanes_screening(df_validated)
        self.assertEqual(res["status"], "success")
        self.assertTrue(0.0 <= res["screening_score"] <= 1.0)

    def test_positive_class_probability_extraction_logic(self):
        """Verify positive class index is dynamically derived from model.classes_."""
        pipeline = load_pipeline("nhanes_screening_core")
        clf = pipeline.named_steps["classifier"]
        self.assertIn(1, clf.classes_)
        expected_pos_idx = int(np.where(clf.classes_ == 1)[0][0])

        is_valid, _, df_validated = validate_nhanes_inputs(self.valid_payload)
        self.assertTrue(is_valid)
        raw_probs = pipeline.predict_proba(df_validated)

        res = predict_nhanes_screening(df_validated)
        self.assertAlmostEqual(res["screening_score"], float(raw_probs[0, expected_pos_idx]), places=4)

    def test_classification_uses_saved_threshold(self):
        """Verify classification threshold matches metadata and determines signal label."""
        metadata = load_model_metadata()
        expected_tau = float(
            metadata["artifacts"]["nhanes_screening_core"]["selected_decision_threshold"]
        )

        is_valid, _, df_validated = validate_nhanes_inputs(self.valid_payload)
        res = predict_nhanes_screening(df_validated)
        self.assertAlmostEqual(res["operating_threshold"], expected_tau, places=4)

        if res["screening_score"] >= expected_tau:
            self.assertTrue(res["is_higher_signal"])
            self.assertEqual(res["classification_label"], "Higher Screening Signal")
        else:
            self.assertFalse(res["is_higher_signal"])
            self.assertEqual(res["classification_label"], "Lower Screening Signal")

    def test_prediction_uses_saved_model_without_retraining(self):
        """Verify running predictions does not touch or modify the model joblib file."""
        models_dir = get_models_dir()
        art_path = models_dir / "nhanes_screening_core.joblib"
        mtime_before = art_path.stat().st_mtime_ns

        is_valid, _, df_validated = validate_nhanes_inputs(self.valid_payload)
        _ = predict_nhanes_screening(df_validated)

        mtime_after = art_path.stat().st_mtime_ns
        self.assertEqual(mtime_before, mtime_after, "Model artifact was modified during prediction (retraining detected)!")

    def test_missing_or_corrupted_artifacts_handled_gracefully(self):
        """Verify predict_nhanes_screening raises clean RuntimeError if artifacts missing."""
        is_valid, _, df_validated = validate_nhanes_inputs(self.valid_payload)
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(RuntimeError) as ctx:
                predict_nhanes_screening(df_validated, models_dir=empty_dir)
            self.assertIn("Failed to load NHANES core model", str(ctx.exception))

    def test_prediction_errors_handled_safely_on_schema_mismatch(self):
        """Verify passing a DataFrame missing required columns raises descriptive ValueError."""
        bad_df = pd.DataFrame([{"age_years": 50, "sex": "male"}])
        with self.assertRaises(ValueError) as ctx:
            predict_nhanes_screening(bad_df)
        self.assertIn("missing required predictor columns", str(ctx.exception))

    def test_unanswered_binary_fields_rejected_without_defaulting_to_no(self):
        """Verify unanswered binary fields (None) fail validation and do not silently default to 0.0."""
        binary_fields = ["bp_medication", "diabetes_diagnosed", "coronary_heart_disease", "stroke"]
        for b_field in binary_fields:
            payload = self.valid_payload.copy()
            payload[b_field] = None  # User left radio unselected
            is_valid, errors, _ = validate_nhanes_inputs(payload)
            self.assertFalse(is_valid, f"Validation should fail when '{b_field}' is None")
            self.assertTrue(
                any(b_field in err for err in errors),
                f"Error message must explicitly mention unselected field '{b_field}': {errors}"
            )

    def test_all_application_pages_import_cleanly(self):
        """Verify all page renderers import and are callable."""
        self.assertTrue(callable(render_nhanes_page))
        self.assertTrue(callable(render_dashboard_page))
        self.assertTrue(callable(render_framingham_page))
        self.assertTrue(callable(render_lipid_page))
        self.assertTrue(callable(render_lifestyle_page))
        self.assertTrue(callable(render_reports_page))


if __name__ == "__main__":
    unittest.main()
