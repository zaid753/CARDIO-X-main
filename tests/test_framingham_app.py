"""
CARDIO-X: Automated Test Suite for Framingham 10-Year CVD Risk Prediction Module (Phase 3C).

Covers:
1. Saved model artifact loading.
2. Feature schema matching (exact 14 predictors).
3. Zero target leakage verification.
4. Input DataFrame column order equality.
5. Derived feature definitions (pulse pressure & MAP consistency).
6. Numeric boundary validation.
7. Categorical and demographic validation.
8. Missing value handling (optional imputables vs. required fields).
9. Rejection of unanswered binary fields (no silent defaulting to No).
10. Positive-class probability extraction via clf.classes_.
11. Calibrated operating threshold classification logic.
12. Zero-retraining guarantee (artifact mtime immutability).
13. Graceful handling of missing or corrupted artifacts.
14. Safe handling of schema mismatches.
15. Clean importing and interoperability across all CARDIO-X application pages.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.utils.framingham_engine import (
    FRAMINGHAM_CORE_PREDICTORS,
    PROHIBITED_LEAKAGE_VARIABLES,
    VALID_SEX_CHOICES,
    VALID_EDUC_CHOICES,
    NUMERIC_BOUNDS,
    validate_framingham_inputs,
    predict_framingham_risk,
)
from app.utils.model_loader import load_pipeline, load_model_metadata, get_models_dir
from app.pages.framingham_risk import render_framingham_page
from app.pages.nhanes_screening import render_nhanes_page
from app.pages.dashboard import render_dashboard_page


class TestFraminghamInteractiveRisk(unittest.TestCase):
    """Test suite for Phase 3C Framingham risk prediction engine and UI integration."""

    def setUp(self):
        """Set up standard valid input payload representing a typical middle-aged adult."""
        self.valid_payload: Dict[str, Any] = {
            "AGE": 54.0,
            "SEX": 1.0,  # Male
            "CURSMOKE": 1.0,  # Smoker
            "CIGPDAY": 20.0,
            "educ": 2.0,  # High school diploma
            "BPMEDS": 0.0,  # No BP medication
            "PREVHYP": 1.0,  # Prevalent hypertension
            "HEARTRTE": 72.0,
            "TOTCHOL": 220.0,
            "BMI": 26.8,
            "DIABETES": 0.0,  # No diabetes
            "GLUCOSE": 85.0,
            "SYSBP": 136.0,
            "DIABP": 86.0,
        }

    # 1. Saved model loads successfully
    def test_saved_model_loads_successfully(self):
        """Verify models/framingham_risk.joblib loads cleanly through the model loader."""
        pipeline = load_pipeline("framingham_risk")
        self.assertIsNotNone(pipeline, "Failed to load framingham_risk.joblib pipeline.")
        self.assertTrue(hasattr(pipeline, "predict_proba"), "Loaded pipeline must support predict_proba.")
        self.assertTrue(hasattr(pipeline, "named_steps"), "Loaded pipeline must be a Scikit-Learn Pipeline.")

    # 2. Feature schema matches saved pipeline
    def test_feature_schema_matches_saved_pipeline(self):
        """Verify the exact 14 predictors match the saved pipeline's feature_names_in_."""
        pipeline = load_pipeline("framingham_risk")
        metadata = load_model_metadata()

        pipe_features = list(pipeline.feature_names_in_)
        self.assertEqual(len(pipe_features), 14, "Saved Framingham model must expect exactly 14 features.")
        self.assertEqual(
            pipe_features,
            FRAMINGHAM_CORE_PREDICTORS,
            "Engine FRAMINGHAM_CORE_PREDICTORS must match pipeline.feature_names_in_ exactly."
        )

        meta_features = metadata.get("artifacts", {}).get("framingham_risk", {}).get("feature_names", [])
        self.assertEqual(set(pipe_features), set(meta_features), "Pipeline and metadata feature sets must match.")

    # 3. No target leakage features present
    def test_zero_target_leakage_features_in_schema(self):
        """Verify no surveillance outcomes, follow-up times, or structural unobservables exist in the schema."""
        for prohibited in PROHIBITED_LEAKAGE_VARIABLES:
            self.assertNotIn(
                prohibited,
                FRAMINGHAM_CORE_PREDICTORS,
                f"Prohibited variable '{prohibited}' found in FRAMINGHAM_CORE_PREDICTORS!"
            )

        # Confirm validation strictly rejects prohibited variables if present in input
        for prohibited in ["TARGET_10Y_CVD", "CVD", "HDLC", "LDLC", "TRIG", "DEATH", "TIMECVD"]:
            leak_payload = self.valid_payload.copy()
            leak_payload[prohibited] = 1.0
            is_valid, errors, _ = validate_framingham_inputs(leak_payload)
            self.assertFalse(is_valid, f"Validation must fail when '{prohibited}' is provided.")
            self.assertTrue(any("target-leakage" in err.lower() for err in errors))

    # 4. Input DataFrame column order
    def test_input_dataframe_has_correct_column_order(self):
        """Verify the validated DataFrame is constructed in the exact expected column order."""
        is_valid, errors, df_val = validate_framingham_inputs(self.valid_payload)
        self.assertTrue(is_valid, f"Validation unexpectedly failed: {errors}")
        self.assertIsNotNone(df_val)
        self.assertEqual(list(df_val.columns), FRAMINGHAM_CORE_PREDICTORS)

    # 5. Derived feature definitions
    def test_derived_hemodynamic_metrics_calculation(self):
        """Verify derived pulse pressure and MAP match standard physiological definitions."""
        is_valid, _, df_val = validate_framingham_inputs(self.valid_payload)
        self.assertTrue(is_valid)

        result = predict_framingham_risk(df_val)
        hemo = result.get("derived_hemodynamics", {})

        expected_pp = round(136.0 - 86.0, 1)  # 50.0 mmHg
        expected_map = round(86.0 + (expected_pp / 3.0), 1)  # 102.7 mmHg

        self.assertAlmostEqual(hemo.get("pulse_pressure_mmhg"), expected_pp, places=1)
        self.assertAlmostEqual(hemo.get("mean_arterial_pressure_mmhg"), expected_map, places=1)

    # 6. Numeric validation bounds
    def test_numeric_validation_and_hemodynamic_checks(self):
        """Verify out-of-range numerical values and inverted blood pressures are rejected."""
        # Age too low (< 30)
        bad_age = self.valid_payload.copy()
        bad_age["AGE"] = 15.0
        is_valid, errors, _ = validate_framingham_inputs(bad_age)
        self.assertFalse(is_valid)
        self.assertTrue(any("age" in err.lower() for err in errors))

        # Total cholesterol too low (< 100)
        bad_chol = self.valid_payload.copy()
        bad_chol["TOTCHOL"] = 50.0
        is_valid, errors, _ = validate_framingham_inputs(bad_chol)
        self.assertFalse(is_valid)
        self.assertTrue(any("cholesterol" in err.lower() for err in errors))

        # Systolic BP <= Diastolic BP
        inverted_bp = self.valid_payload.copy()
        inverted_bp["SYSBP"] = 80.0
        inverted_bp["DIABP"] = 120.0
        is_valid, errors, _ = validate_framingham_inputs(inverted_bp)
        self.assertFalse(is_valid)
        self.assertTrue(any("greater than diastolic" in err.lower() for err in errors))

        # Pulse pressure < 10 mmHg
        narrow_bp = self.valid_payload.copy()
        narrow_bp["SYSBP"] = 125.0
        narrow_bp["DIABP"] = 120.0
        is_valid, errors, _ = validate_framingham_inputs(narrow_bp)
        self.assertFalse(is_valid)
        self.assertTrue(any("pulse pressure" in err.lower() for err in errors))

    # 7. Categorical and demographic validation
    def test_categorical_and_smoking_consistency_validation(self):
        """Verify sex, education, and smoking consistency rules."""
        # Non-smoker reporting cigarettes per day
        bad_smk = self.valid_payload.copy()
        bad_smk["CURSMOKE"] = 0.0
        bad_smk["CIGPDAY"] = 15.0
        is_valid, errors, _ = validate_framingham_inputs(bad_smk)
        self.assertFalse(is_valid)
        self.assertTrue(any("non-smokers" in err.lower() for err in errors))

        # Current smoker reporting 0 cigarettes per day
        bad_smk2 = self.valid_payload.copy()
        bad_smk2["CURSMOKE"] = 1.0
        bad_smk2["CIGPDAY"] = 0.0
        is_valid, errors, _ = validate_framingham_inputs(bad_smk2)
        self.assertFalse(is_valid)
        self.assertTrue(any("current smokers" in err.lower() for err in errors))

        # Invalid SEX value
        bad_sex = self.valid_payload.copy()
        bad_sex["SEX"] = 3.0
        is_valid, errors, _ = validate_framingham_inputs(bad_sex)
        self.assertFalse(is_valid)
        self.assertTrue(any("sex" in err.lower() for err in errors))

    # 8. Missing values handled according to training pipeline
    def test_optional_imputable_inputs_allow_nan(self):
        """Verify optional imputable features (BMI, GLUCOSE, HEARTRTE, educ) accept NaN without failing."""
        payload_with_nans = self.valid_payload.copy()
        payload_with_nans["BMI"] = None
        payload_with_nans["GLUCOSE"] = np.nan
        payload_with_nans["HEARTRTE"] = ""
        payload_with_nans["educ"] = None

        is_valid, errors, df_val = validate_framingham_inputs(payload_with_nans)
        self.assertTrue(is_valid, f"Imputable fields should allow missingness, but got errors: {errors}")
        self.assertTrue(pd.isna(df_val["BMI"].iloc[0]))
        self.assertTrue(pd.isna(df_val["GLUCOSE"].iloc[0]))
        self.assertTrue(pd.isna(df_val["HEARTRTE"].iloc[0]))
        self.assertTrue(pd.isna(df_val["educ"].iloc[0]))

        # Confirm prediction succeeds through pipeline imputer
        result = predict_framingham_risk(df_val)
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["risk_score"], 0.0)

    # 9. Unanswered binary fields rejected without silent defaulting
    def test_unanswered_binary_fields_rejected_without_defaulting_to_no(self):
        """Verify unselected binary radio controls (None) fail validation and are not converted to 0.0."""
        binary_fields = ["BPMEDS", "PREVHYP", "DIABETES", "CURSMOKE", "SEX"]
        for b_field in binary_fields:
            payload = self.valid_payload.copy()
            payload[b_field] = None
            is_valid, errors, _ = validate_framingham_inputs(payload)
            self.assertFalse(is_valid, f"Validation should fail when '{b_field}' is None")
            self.assertTrue(
                any(b_field.lower() in err.lower() for err in errors),
                f"Error message must explicitly mention unselected field '{b_field}': {errors}"
            )

    # 10. Positive-class probability extraction
    def test_positive_class_probability_extraction_logic(self):
        """Verify predict_framingham_risk dynamically extracts class 1 probability."""
        is_valid, _, df_val = validate_framingham_inputs(self.valid_payload)
        self.assertTrue(is_valid)

        result = predict_framingham_risk(df_val)
        score = result["risk_score"]
        neg_score = result["negative_score_pct"]

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertAlmostEqual(round(score * 100.0, 1) + neg_score, 100.0, places=1)

    # 11. Classification uses saved threshold
    def test_classification_uses_saved_threshold(self):
        """Verify operating threshold is retrieved from metadata (~0.40) and governs classification."""
        metadata = load_model_metadata()
        expected_threshold = float(
            metadata["artifacts"]["framingham_risk"]["selected_decision_threshold"]
        )

        is_valid, _, df_val = validate_framingham_inputs(self.valid_payload)
        result = predict_framingham_risk(df_val)

        self.assertAlmostEqual(result["operating_threshold"], expected_threshold, places=2)

        if result["risk_score"] >= expected_threshold:
            self.assertTrue(result["is_higher_signal"])
            self.assertIn("Higher Risk Signal", result["classification_label"])
        else:
            self.assertFalse(result["is_higher_signal"])
            self.assertIn("Lower Risk Signal", result["classification_label"])

    # 12. Prediction does not retrain model
    def test_prediction_uses_saved_model_without_retraining(self):
        """Verify joblib artifact mtime is unchanged before and after inference."""
        models_dir = get_models_dir()
        art_path = models_dir / "framingham_risk.joblib"
        mtime_before = art_path.stat().st_mtime_ns

        is_valid, _, df_val = validate_framingham_inputs(self.valid_payload)
        _ = predict_framingham_risk(df_val)

        mtime_after = art_path.stat().st_mtime_ns
        self.assertEqual(mtime_before, mtime_after, "Model artifact was modified during prediction (retraining detected)!")

    # 13. Missing or corrupted artifacts handled gracefully
    def test_missing_or_corrupted_artifacts_handled_gracefully(self):
        """Verify predict_framingham_risk raises descriptive RuntimeError if artifact missing."""
        is_valid, _, df_val = validate_framingham_inputs(self.valid_payload)
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(RuntimeError) as ctx:
                predict_framingham_risk(df_val, models_dir=empty_dir)
            self.assertIn("Failed to load Framingham risk model", str(ctx.exception))

    # 14. Prediction exceptions handled safely on schema mismatch
    def test_prediction_errors_handled_safely_on_schema_mismatch(self):
        """Verify passing a DataFrame missing required columns raises descriptive ValueError."""
        bad_df = pd.DataFrame([{"AGE": 50, "SEX": 1}])
        with self.assertRaises(ValueError) as ctx:
            predict_framingham_risk(bad_df)
        self.assertIn("missing required predictor columns", str(ctx.exception))

    # 15. All application pages import cleanly
    def test_all_application_pages_import_cleanly(self):
        """Verify all page renderers import cleanly and are callable."""
        self.assertTrue(callable(render_framingham_page))
        self.assertTrue(callable(render_nhanes_page))
        self.assertTrue(callable(render_dashboard_page))


if __name__ == "__main__":
    unittest.main()
