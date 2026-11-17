"""
Unit tests for Cardio-X NHANES Preprocessing Module (CARDIO-X Model 1).
Validates Target Option B construction, indeterminate exclusion, zero target leakage,
feature selection, stratified splitting, preprocessor fitting isolation,
and source file immutability.
"""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.nhanes_preprocessing import (
    AUDITED_NON_INVASIVE_PREDICTORS,
    RECOMMENDED_CORE_PREDICTORS,
    EXCLUDED_VARIABLES,
    SURVEY_METADATA_COLUMNS,
    CONTINUOUS_PREDICTORS,
    BINARY_ORDINAL_PREDICTORS,
    CATEGORICAL_PREDICTORS,
    load_nhanes_dataset,
    validate_nhanes_columns,
    build_nhanes_screening_cohort,
    extract_nhanes_feature_sets,
    get_nhanes_feature_types,
    create_nhanes_preprocessor,
    split_nhanes_dataset,
    run_nhanes_preprocessing_pipeline,
)


class TestNHANESPreprocessing(unittest.TestCase):
    """Comprehensive test suite for NHANES screening preprocessing pipeline."""

    @classmethod
    def setUpClass(cls):
        """Load real NHANES dataset if available on disk."""
        cls.data_path = Path("data_comparison/nhanes_diabetes_2021_2023.csv")
        cls.has_real_data = cls.data_path.is_file()
        if cls.has_real_data:
            cls.df_raw = pd.read_csv(cls.data_path)

    def setUp(self):
        """Create a deterministic synthetic test dataframe covering all target boundary cases."""
        self.mock_df = pd.DataFrame({
            "participant_id": [101, 102, 103, 104, 105, 106, 107, 108, 109],
            "survey_cycle": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "survey_weight_interview_2yr": [1000.0] * 9,
            "survey_weight_mec_2yr": [1100.0] * 9,
            "survey_weight_fasting_2yr": [1200.0] * 9,
            "survey_strata": [1] * 9,
            "survey_psu": [1] * 9,
            # Target components
            "total_cholesterol_mg_dl": [
                260.0,   # Row 0: tc >= 240, med == 0 -> POSITIVE (1)
                180.0,   # Row 1: tc < 240, med == 1  -> POSITIVE (1)
                270.0,   # Row 2: tc >= 240, med == 1 -> POSITIVE (1)
                250.0,   # Row 3: tc >= 240, med NaN  -> POSITIVE (1)
                np.nan,  # Row 4: tc NaN, med == 1    -> POSITIVE (1)
                190.0,   # Row 5: tc < 240, med == 0  -> NEGATIVE (0)
                np.nan,  # Row 6: tc NaN, med == 0    -> INDETERMINATE
                195.0,   # Row 7: tc < 240, med NaN   -> INDETERMINATE
                np.nan,  # Row 8: tc NaN, med NaN     -> INDETERMINATE
            ],
            "cholesterol_medication": [
                0.0, 1.0, 1.0, np.nan, 1.0, 0.0, 0.0, np.nan, np.nan
            ],
            # Excluded leakage components
            "high_cholesterol_diagnosed": [1, 1, 1, 1, 1, 0, 0, 0, 0],
            "non_hdl_cholesterol_mg_dl": [180.0] * 9,
            "ldl_cholesterol_mg_dl": [150.0] * 9,
            "triglycerides_mg_dl": [160.0] * 9,
            "triglyceride_hdl_ratio": [3.0] * 9,
            "hdl_cholesterol_mg_dl": [50.0] * 9,
            "diabetes_undiagnosed": [0] * 9,
            "glycemic_status": [0] * 9,
            "alt_u_l": [20.0] * 9,
            "ast_u_l": [22.0] * 9,
            "ggt_u_l": [18.0] * 9,
            "uric_acid_mg_dl": [5.0] * 9,
            "creatinine_mg_dl": [0.9] * 9,
            "albumin_g_dl": [4.2] * 9,
            "egfr_ml_min_1_73m2": [95.0] * 9,
            "white_blood_cell_count_1000_ul": [6.5] * 9,
            "hemoglobin_g_dl": [14.0] * 9,
            "platelet_count_1000_ul": [250.0] * 9,
            "age_group": [1] * 9,
            "education_code": [2] * 9,
            "bmi_category": [2] * 9,
            # Continuous predictors
            "age_years": [45, 52, 60, 38, 55, 42, 48, 50, 62],
            "family_income_poverty_ratio": [2.5, 3.1, 1.2, 4.0, 2.0, 3.5, 1.8, 2.2, 3.0],
            "bmi_kg_m2": [24.5, 28.2, 31.0, 22.0, 29.5, 26.0, 27.5, 30.0, 25.0],
            "weight_kg": [70.0, 80.0, 90.0, 65.0, 85.0, 75.0, 78.0, 88.0, 72.0],
            "height_cm": [170.0, 168.0, 172.0, 175.0, 169.0, 171.0, 167.0, 173.0, 170.0],
            "waist_circumference_cm": [82.0, 92.0, 102.0, 78.0, 98.0, 86.0, 90.0, 100.0, 84.0],
            "hip_circumference_cm": [95.0, 102.0, 110.0, 92.0, 106.0, 98.0, 100.0, 108.0, 96.0],
            "arm_circumference_cm": [30.0, 32.0, 35.0, 28.0, 33.0, 31.0, 31.5, 34.0, 29.5],
            "waist_to_height_ratio": [0.48, 0.55, 0.59, 0.45, 0.58, 0.50, 0.54, 0.58, 0.49],
            "waist_to_hip_ratio": [0.86, 0.90, 0.93, 0.85, 0.92, 0.88, 0.90, 0.93, 0.88],
            "weight_change_1yr_kg": [0.0, 2.0, -1.5, 1.0, 0.5, -2.0, 3.0, 1.5, 0.0],
            "avg_systolic_bp_mmhg": [120.0, 135.0, 145.0, 115.0, 140.0, 122.0, 130.0, 138.0, 125.0],
            "avg_diastolic_bp_mmhg": [80.0, 85.0, 90.0, 75.0, 88.0, 82.0, 84.0, 86.0, 78.0],
            "avg_pulse_bpm": [72.0, 76.0, 80.0, 68.0, 78.0, 70.0, 74.0, 82.0, 69.0],
            "pulse_pressure_mmhg": [40.0, 50.0, 55.0, 40.0, 52.0, 40.0, 46.0, 52.0, 47.0],
            "moderate_ltp_minutes_week": [150.0, 60.0, 0.0, 200.0, 90.0, 120.0, 30.0, 0.0, 180.0],
            "vigorous_ltp_minutes_week": [60.0, 0.0, 0.0, 90.0, 30.0, 45.0, 0.0, 0.0, 60.0],
            "total_ltp_met_minutes_week": [1200.0, 400.0, 0.0, 1800.0, 700.0, 1000.0, 200.0, 0.0, 1500.0],
            "sedentary_minutes_day": [360.0, 480.0, 600.0, 300.0, 540.0, 420.0, 500.0, 580.0, 350.0],
            "alcohol_drinks_per_week": [2.0, 5.0, 0.0, 1.0, 8.0, 3.0, 0.0, 4.0, 2.0],
            "sleep_hours_weekday": [7.5, 6.0, 5.5, 8.0, 6.5, 7.0, 6.0, 5.0, 7.5],
            "sleep_hours_weekend": [8.0, 7.0, 6.5, 8.5, 7.5, 8.0, 7.0, 6.5, 8.0],
            "phq9_score": [2.0, 6.0, 12.0, 1.0, 8.0, 3.0, 5.0, 10.0, 2.0],
            # Binary & Ordinal predictors
            "born_in_us": [1, 1, 0, 1, 1, 1, 0, 1, 1],
            "income_poverty_category": [2, 3, 1, 4, 2, 3, 1, 2, 3],
            "pregnant": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "tried_to_lose_weight_past_year": [0, 1, 1, 0, 1, 0, 1, 1, 0],
            "hypertension_measured": [0, 1, 1, 0, 1, 0, 0, 1, 0],
            "hypertension_diagnosed": [0, 1, 1, 0, 1, 0, 0, 1, 0],
            "bp_medication": [0, 1, 1, 0, 1, 0, 0, 1, 0],
            "meets_activity_guidelines": [1, 0, 0, 1, 0, 1, 0, 0, 1],
            "smoked_100_cigarettes": [0, 1, 1, 0, 1, 0, 1, 1, 0],
            "heavy_alcohol_use": [0, 0, 0, 0, 1, 0, 0, 0, 0],
            "depression_screen_positive": [0, 0, 1, 0, 0, 0, 0, 1, 0],
            "general_health": [2, 3, 4, 1, 3, 2, 3, 4, 2],
            "has_health_insurance": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "routine_healthcare_place": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "mental_health_care_past_year": [0, 0, 1, 0, 0, 0, 0, 1, 0],
            "asthma": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "congestive_heart_failure": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "coronary_heart_disease": [0, 0, 1, 0, 0, 0, 0, 0, 0],
            "angina": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "heart_attack": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "stroke": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "liver_condition": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "thyroid_problem": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "cancer_any": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "weak_or_failing_kidneys": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "heart_disease_or_attack": [0, 0, 1, 0, 0, 0, 0, 0, 0],
            "diabetes": [0, 0, 1, 0, 0, 0, 0, 1, 0],
            "diabetes_diagnosed": [0, 0, 1, 0, 0, 0, 0, 1, 0],
            # Categorical string predictors
            "sex": ["male", "female", "male", "female", "male", "female", "male", "female", "male"],
            "race_ethnicity": [
                "Non-Hispanic White", "Non-Hispanic Black", "Mexican American",
                "Other Hispanic", "Non-Hispanic Asian", "Non-Hispanic White",
                "Other or multiracial", "Non-Hispanic Black", "Non-Hispanic White"
            ],
            "education_level": [
                "College graduate or above", "Some college/AA", "High school/GED",
                "College graduate or above", "Some college/AA", "High school/GED",
                "9th-11th grade", "Some college/AA", "College graduate or above"
            ],
            "marital_status": [
                "Married or living with partner", "Never married", "Widowed, divorced or separated",
                "Married or living with partner", "Never married", "Married or living with partner",
                "Widowed, divorced or separated", "Married or living with partner", "Never married"
            ],
            "work_status": [
                "Working", "Not working", "Working", "Working",
                "Not working", "Working", "Looking for work", "Working", "Not working"
            ],
            "smoking_status": [
                "never", "former", "current", "never",
                "former", "never", "former", "current", "never"
            ],
            "general_health_label": [
                "Very good", "Good", "Fair", "Excellent",
                "Good", "Very good", "Good", "Fair", "Very good"
            ],
        })

    def test_synthetic_target_construction_and_indeterminate_handling(self):
        """Verify Option B logic across all combinations of total cholesterol and medication."""
        eval_df, audit = build_nhanes_screening_cohort(self.mock_df, return_audit=True)

        # 9 raw rows: 5 positive (rows 0, 1, 2, 3, 4), 1 negative (row 5), 3 indeterminate (rows 6, 7, 8)
        self.assertEqual(audit["total_raw_rows"], 9)
        self.assertEqual(audit["positive_cases"], 5)
        self.assertEqual(audit["negative_cases"], 1)
        self.assertEqual(audit["indeterminate_cases"], 3)
        self.assertEqual(audit["evaluable_rows"], 6)

        # Verify excluded participants are 107, 108, 109
        self.assertNotIn(107, eval_df["participant_id"].values)
        self.assertNotIn(108, eval_df["participant_id"].values)
        self.assertNotIn(109, eval_df["participant_id"].values)

        # Verify positive participants are 101-105
        for pid in [101, 102, 103, 104, 105]:
            self.assertEqual(eval_df.loc[eval_df["participant_id"] == pid, "TARGET_HYPERCHOL"].values[0], 1)

        # Verify negative participant is 106
        self.assertEqual(eval_df.loc[eval_df["participant_id"] == 106, "TARGET_HYPERCHOL"].values[0], 0)

    def test_actual_dataset_cohort_counts(self):
        """Verify exact counts from the audit against the actual NHANES 2021-2023 dataset."""
        if not self.has_real_data:
            self.skipTest("Raw NHANES dataset not found.")

        eval_df, audit = build_nhanes_screening_cohort(self.df_raw, return_audit=True)

        self.assertEqual(audit["total_raw_rows"], 6001)
        self.assertEqual(audit["positive_cases"], 2196)
        self.assertEqual(audit["negative_cases"], 3578)
        self.assertEqual(audit["indeterminate_cases"], 227)
        self.assertEqual(audit["evaluable_rows"], 5774)
        self.assertEqual(audit["prevalence_pct"], 38.03)

        # Check indeterminate breakdown
        self.assertEqual(audit["indeterminate_breakdown"]["tc_null_unmedicated"], 202)
        self.assertEqual(audit["indeterminate_breakdown"]["med_null_tc_normal"], 23)
        self.assertEqual(audit["indeterminate_breakdown"]["both_null"], 2)

        # Verify target column is binary
        self.assertEqual(set(eval_df["TARGET_HYPERCHOL"].unique()), {0, 1})
        self.assertEqual(int(eval_df["TARGET_HYPERCHOL"].sum()), 2196)

    def test_zero_target_leakage_in_features(self):
        """Verify that predictor matrices strictly exclude target definers, proxies, and invasive labs."""
        eval_df, _ = build_nhanes_screening_cohort(self.mock_df, return_audit=True)

        for subset_name in ["all_58", "core_24"]:
            X, y, survey_meta = extract_nhanes_feature_sets(eval_df, feature_subset=subset_name)

            # Check no excluded variable is present
            for col in X.columns:
                self.assertNotIn(col, EXCLUDED_VARIABLES, f"Leakage: Excluded variable '{col}' in '{subset_name}'")
                self.assertNotIn("cholesterol", col.lower(), f"Leakage: Cholesterol column '{col}' in '{subset_name}'")
                self.assertNotIn("ldl", col.lower(), f"Leakage: LDL column '{col}' in '{subset_name}'")
                self.assertNotIn("trig", col.lower(), f"Leakage: Triglyceride column '{col}' in '{subset_name}'")
                self.assertNotIn("TARGET", col, f"Leakage: Target column '{col}' in '{subset_name}'")
                self.assertNotIn("survey_", col, f"Leakage: Survey design column '{col}' in '{subset_name}'")

            # Check y is strictly binary
            self.assertTrue(set(y.unique()).issubset({0, 1}))

            # Check survey metadata is preserved separately
            self.assertIn("survey_weight_mec_2yr", survey_meta.columns)
            self.assertIn("survey_strata", survey_meta.columns)
            self.assertIn("survey_psu", survey_meta.columns)

    def test_non_invasive_feature_selection_counts(self):
        """Verify exact counts for full 58-feature set and 24-core curated subset."""
        eval_df, _ = build_nhanes_screening_cohort(self.mock_df, return_audit=True)

        X_all, _, _ = extract_nhanes_feature_sets(eval_df, feature_subset="all_58")
        self.assertEqual(X_all.shape[1], 58)
        self.assertEqual(list(X_all.columns), AUDITED_NON_INVASIVE_PREDICTORS)

        X_core, _, _ = extract_nhanes_feature_sets(eval_df, feature_subset="core_24")
        self.assertEqual(X_core.shape[1], 24)
        self.assertEqual(list(X_core.columns), RECOMMENDED_CORE_PREDICTORS)

        # Core subset must be a strict subset of all 58
        self.assertTrue(set(X_core.columns).issubset(set(X_all.columns)))

    def test_feature_type_partitioning(self):
        """Verify that get_nhanes_feature_types correctly classifies continuous, binary/ordinal, and categorical."""
        eval_df, _ = build_nhanes_screening_cohort(self.mock_df, return_audit=True)
        X_all, _, _ = extract_nhanes_feature_sets(eval_df, feature_subset="all_58")

        types_dict = get_nhanes_feature_types(X_all)
        self.assertEqual(len(types_dict["continuous"]), 23)
        self.assertEqual(len(types_dict["binary_ordinal"]), 28)
        self.assertEqual(len(types_dict["categorical"]), 7)
        self.assertEqual(
            len(types_dict["continuous"]) + len(types_dict["binary_ordinal"]) + len(types_dict["categorical"]),
            58
        )

    def test_stratified_train_test_split_isolation(self):
        """Verify stratified split balance and index alignment on the real dataset."""
        if not self.has_real_data:
            self.skipTest("Raw NHANES dataset not found.")

        eval_df, _ = build_nhanes_screening_cohort(self.df_raw, return_audit=True)
        X, y, survey_meta = extract_nhanes_feature_sets(eval_df, feature_subset="all_58")

        X_train, X_test, y_train, y_test, survey_train, survey_test = split_nhanes_dataset(
            X, y, survey_meta=survey_meta, test_size=0.20, random_state=42
        )

        # Check exact partition sizes (N=5,774 -> 4,619 Train, 1,155 Test)
        self.assertEqual(len(X_train), 4619)
        self.assertEqual(len(X_test), 1155)
        self.assertEqual(len(X_train) + len(X_test), 5774)

        # Check stratification balance
        self.assertEqual(int(y_train.sum()), 1757)
        self.assertEqual(int(y_test.sum()), 439)
        self.assertAlmostEqual(y_train.mean(), 0.3804, places=3)
        self.assertAlmostEqual(y_test.mean(), 0.3801, places=3)

        # Check zero index overlap
        self.assertEqual(len(set(X_train.index).intersection(set(X_test.index))), 0)

        # Check survey metadata index alignment
        self.assertTrue((X_train.index == survey_train.index).all())
        self.assertTrue((X_test.index == survey_test.index).all())

    def test_preprocessor_fitting_isolation(self):
        """Verify preprocessor parameters are computed strictly from X_train without test leakage."""
        if not self.has_real_data:
            self.skipTest("Raw NHANES dataset not found.")

        eval_df, _ = build_nhanes_screening_cohort(self.df_raw, return_audit=True)
        X, y, _ = extract_nhanes_feature_sets(eval_df, feature_subset="all_58")
        X_train, X_test, y_train, y_test, _, _ = split_nhanes_dataset(X, y, random_state=42)

        # Linear model preprocessor
        preprocessor = create_nhanes_preprocessor(feature_subset="all_58", model_family="linear")
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)

        # Assert no NaNs remain after transformation
        self.assertFalse(np.isnan(X_train_trans).any(), "NaN in transformed train array")
        self.assertFalse(np.isnan(X_test_trans).any(), "NaN in transformed test array")

        # Verify imputer median matches X_train median, not combined X median
        cont_imputer = preprocessor.named_transformers_["cont"].named_steps["imputer"]
        expected_train_medians = X_train[CONTINUOUS_PREDICTORS].median().values
        np.testing.assert_allclose(cont_imputer.statistics_, expected_train_medians, rtol=1e-5)

        # Check transformed output shapes (58 raw features expand to 79 one-hot numeric columns)
        self.assertEqual(X_train_trans.shape, (4619, 79))
        self.assertEqual(X_test_trans.shape, (1155, 79))

    def test_tree_vs_linear_scaling_behavior(self):
        """Verify tree family preserves natural units while linear family scales."""
        if not self.has_real_data:
            self.skipTest("Raw NHANES dataset not found.")

        eval_df, _ = build_nhanes_screening_cohort(self.df_raw, return_audit=True)
        X, y, _ = extract_nhanes_feature_sets(eval_df, feature_subset="core_24")
        X_train, _, _, _, _, _ = split_nhanes_dataset(X, y, random_state=42)

        # Linear preprocessor scales
        pipe_linear = create_nhanes_preprocessor(feature_subset="core_24", model_family="linear")
        X_lin = pipe_linear.fit_transform(X_train)

        # Tree preprocessor does not scale
        pipe_tree = create_nhanes_preprocessor(feature_subset="core_24", model_family="tree")
        X_tr = pipe_tree.fit_transform(X_train)

        # In tree preprocessor, age_years (column 0) should be unscaled (e.g. median ~ 50)
        # In linear preprocessor, age_years (column 0) should be scaled around 0
        self.assertTrue(np.median(X_tr[:, 0]) > 20.0, "Tree preprocessor unexpectedly scaled age_years")
        self.assertAlmostEqual(np.median(X_lin[:, 0]), 0.0, places=1)

    def test_source_dataset_immutability(self):
        """Verify that executing run_nhanes_preprocessing_pipeline preserves raw CSV hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_nhanes.csv"

            # Replicate mock data so StratifiedShuffleSplit has sufficient class instances
            test_mock = pd.concat([self.mock_df] * 5, ignore_index=True)
            test_mock["participant_id"] = range(1, len(test_mock) + 1)
            test_mock.to_csv(csv_path, index=False)

            with open(csv_path, "rb") as f:
                initial_hash = hashlib.sha256(f.read()).hexdigest()

            result = run_nhanes_preprocessing_pipeline(
                input_path=str(csv_path),
                feature_subset="all_58",
                assert_expected_counts=False
            )

            self.assertEqual(result["status"], "success")
            self.assertTrue(result["input_file_unchanged"])

            with open(csv_path, "rb") as f:
                final_hash = hashlib.sha256(f.read()).hexdigest()

            self.assertEqual(initial_hash, final_hash)


if __name__ == "__main__":
    unittest.main()
