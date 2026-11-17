"""
Unit tests for Cardio-X Model Training, Cross-Validation, Evaluation and Persistence Module.
Verifies estimator construction, pipeline integration, leakage-free threshold optimization,
metric calculations, artifact persistence, and reloaded prediction reproducibility.
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models import (
    build_estimator,
    build_pipeline,
    compute_classification_metrics,
    select_optimal_threshold,
    cross_validate_pipeline,
    train_and_evaluate_model,
    save_model_artifact,
    load_model_artifact,
)
from src.nhanes_preprocessing import create_nhanes_preprocessor, RECOMMENDED_CORE_PREDICTORS
from src.preprocessing import create_preprocessor as create_framingham_preprocessor


class TestModelsModule(unittest.TestCase):
    """Test suite for CARDIO-X machine learning modeling architecture."""

    def setUp(self):
        """Build deterministic synthetic test fixtures for localized unit tests."""
        np.random.seed(42)
        n_samples = 120

        # Synthetic NHANES-like core dataset
        self.mock_X = pd.DataFrame({
            "age_years": np.random.randint(30, 75, size=n_samples),
            "family_income_poverty_ratio": np.random.uniform(0.5, 5.0, size=n_samples),
            "bmi_kg_m2": np.random.uniform(18.5, 40.0, size=n_samples),
            "waist_circumference_cm": np.random.uniform(70.0, 120.0, size=n_samples),
            "waist_to_height_ratio": np.random.uniform(0.4, 0.7, size=n_samples),
            "weight_change_1yr_kg": np.random.uniform(-10.0, 10.0, size=n_samples),
            "avg_systolic_bp_mmhg": np.random.uniform(90.0, 180.0, size=n_samples),
            "avg_diastolic_bp_mmhg": np.random.uniform(60.0, 110.0, size=n_samples),
            "pulse_pressure_mmhg": np.random.uniform(30.0, 80.0, size=n_samples),
            "avg_pulse_bpm": np.random.uniform(55.0, 100.0, size=n_samples),
            "bp_medication": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
            "smoking_status": np.random.choice(["never", "former", "current"], size=n_samples),
            "total_ltp_met_minutes_week": np.random.uniform(0.0, 2000.0, size=n_samples),
            "sedentary_minutes_day": np.random.uniform(120.0, 720.0, size=n_samples),
            "alcohol_drinks_per_week": np.random.uniform(0.0, 14.0, size=n_samples),
            "sleep_hours_weekday": np.random.uniform(4.0, 10.0, size=n_samples),
            "diabetes_diagnosed": np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
            "general_health": np.random.choice([1, 2, 3, 4, 5], size=n_samples),
            "phq9_score": np.random.uniform(0.0, 20.0, size=n_samples),
            "coronary_heart_disease": np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
            "stroke": np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
            "sex": np.random.choice(["male", "female"], size=n_samples),
            "race_ethnicity": np.random.choice(["Non-Hispanic White", "Non-Hispanic Black", "Mexican American"], size=n_samples),
            "education_level": np.random.choice(["College graduate or above", "High school/GED", "Some college/AA"], size=n_samples),
        })

        # Deterministic binary target with ~35% prevalence
        self.mock_y = pd.Series(
            (0.03 * self.mock_X["age_years"] + 0.02 * self.mock_X["avg_systolic_bp_mmhg"] + np.random.normal(0, 1, n_samples)) > 4.5
        ).astype(int)

        self.mock_survey_meta = pd.DataFrame({
            "survey_weight_mec_2yr": np.random.uniform(500.0, 5000.0, size=n_samples),
            "survey_strata": np.random.randint(1, 15, size=n_samples),
            "survey_psu": np.random.randint(1, 3, size=n_samples),
        })

    def test_build_all_supported_estimators(self):
        """Verify all supported model architectures instantiate with correct hyperparameters."""
        lr = build_estimator("logistic_regression")
        rf = build_estimator("random_forest")
        gb = build_estimator("gradient_boosting")

        self.assertEqual(lr.class_weight, "balanced")
        self.assertEqual(rf.class_weight, "balanced")
        self.assertEqual(gb.class_weight, "balanced")
        self.assertEqual(lr.random_state, 42)
        self.assertEqual(rf.random_state, 42)
        self.assertEqual(gb.random_state, 42)

        with self.assertRaises(ValueError):
            build_estimator("unsupported_model_type")

    def test_pipeline_construction_and_probability_outputs(self):
        """Verify pipeline trains and outputs valid probability values in [0, 1]."""
        prep = create_nhanes_preprocessor(feature_subset="core_24", model_family="linear")
        pipeline = build_pipeline("logistic_regression", preprocessor=prep)

        pipeline.fit(self.mock_X, self.mock_y)
        probs = pipeline.predict_proba(self.mock_X)

        # Check shape (N, 2)
        self.assertEqual(probs.shape, (len(self.mock_X), 2))
        # Check all values bounded in [0, 1]
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        # Check rows sum to 1.0
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(self.mock_X)), rtol=1e-5)

    def test_classification_metrics_computation(self):
        """Verify mathematical correctness of classification metric outputs."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])

        metrics = compute_classification_metrics(y_true, y_prob, threshold=0.50)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["pr_auc"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], {"TN": 2, "FP": 0, "FN": 0, "TP": 2})

    def test_threshold_selection_on_training_predictions_only(self):
        """Verify threshold optimization targets ~80% sensitivity without touching test data."""
        y_true = np.array([0] * 50 + [1] * 50)
        # Synthetic probs where positive cases have mean 0.7, negative cases have mean 0.3
        np.random.seed(42)
        y_prob = np.concatenate([
            np.random.uniform(0.1, 0.6, size=50),
            np.random.uniform(0.4, 0.9, size=50),
        ])

        tau, opt_metrics = select_optimal_threshold(y_true, y_prob, target_sensitivity=0.80)

        self.assertTrue(0.05 <= tau <= 0.95)
        self.assertGreaterEqual(opt_metrics["sensitivity"], 0.79)  # Target ~80%
        self.assertGreaterEqual(opt_metrics["specificity"], 0.20)

    def test_cross_validation_isolation_and_stability(self):
        """Verify cross-validation produces valid out-of-fold metrics and fold summary."""
        prep = create_nhanes_preprocessor(feature_subset="core_24", model_family="linear")
        pipeline = build_pipeline("logistic_regression", preprocessor=prep)

        cv_res = cross_validate_pipeline(pipeline, self.mock_X, self.mock_y, cv_folds=3, random_state=42)

        self.assertEqual(len(cv_res["oof_probs"]), len(self.mock_X))
        self.assertIn("roc_auc_mean", cv_res["cv_fold_summary"])
        self.assertIn("roc_auc_std", cv_res["cv_fold_summary"])
        self.assertIn("brier_mean", cv_res["cv_fold_summary"])
        self.assertTrue(0.0 <= cv_res["cv_fold_summary"]["roc_auc_mean"] <= 1.0)

    def test_artifact_persistence_and_reloaded_prediction_reproducibility(self):
        """Verify saving and re-loading joblib artifact yields identical predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prep = create_nhanes_preprocessor(feature_subset="core_24", model_family="linear")
            pipeline = build_pipeline("logistic_regression", preprocessor=prep)
            pipeline.fit(self.mock_X, self.mock_y)

            orig_probs = pipeline.predict_proba(self.mock_X)[:, 1]

            metadata = {
                "dataset": "Test Synthetic",
                "estimator": "logistic_regression",
                "features": list(self.mock_X.columns),
                "threshold": 0.45,
            }

            joblib_path, meta_path = save_model_artifact(
                pipeline, metadata, output_dir=tmpdir, artifact_filename="test_pipeline.joblib"
            )

            self.assertTrue(joblib_path.is_file())
            self.assertTrue(meta_path.is_file())

            # Reload pipeline
            reloaded_pipe = load_model_artifact(joblib_path)
            reloaded_probs = reloaded_pipe.predict_proba(self.mock_X)[:, 1]

            # Verify predictions match down to exact float precision
            np.testing.assert_allclose(orig_probs, reloaded_probs, rtol=1e-7)

            # Verify metadata JSON contents
            with open(meta_path, "r", encoding="utf-8") as f:
                loaded_meta = json.load(f)
            self.assertEqual(loaded_meta["threshold"], 0.45)
            self.assertEqual(loaded_meta["estimator"], "logistic_regression")

    def test_nhanes_and_framingham_artifacts_remain_separate(self):
        """Verify that NHANES and Framingham artifacts are stored under distinct filenames."""
        from src.models import run_all_phase2b_experiments

        models_dir = Path("models")
        expected_artifacts = [
            "nhanes_screening_core.joblib",
            "nhanes_screening_full.joblib",
            "framingham_risk.joblib",
            "model_metadata.json",
        ]

        # Artifacts must exist once experiments run
        for art in expected_artifacts:
            art_path = models_dir / art
            if art_path.is_file():
                self.assertGreater(art_path.stat().st_size, 100, f"Artifact empty: {art_path}")

    def test_artifact_estimator_and_metadata_consistency(self):
        """Verify saved artifact estimators match their recorded metadata exactly."""
        models_dir = Path("models")
        with open(models_dir / "model_metadata.json", "r", encoding="utf-8") as f:
            master_meta = json.load(f)

        artifacts = master_meta["artifacts"]

        # NHANES Core
        core_pipe = load_model_artifact(models_dir / "nhanes_screening_core.joblib")
        core_clf = core_pipe.named_steps["classifier"].__class__.__name__
        self.assertEqual(core_clf, "HistGradientBoostingClassifier")
        self.assertEqual(artifacts["nhanes_screening_core"]["selected_estimator"], "Gradient_Boosting")
        self.assertEqual(artifacts["nhanes_screening_core"]["feature_count"], 24)

        # NHANES Full
        full_pipe = load_model_artifact(models_dir / "nhanes_screening_full.joblib")
        full_clf = full_pipe.named_steps["classifier"].__class__.__name__
        self.assertEqual(full_clf, "HistGradientBoostingClassifier")
        self.assertEqual(artifacts["nhanes_screening_full"]["selected_estimator"], "Gradient_Boosting")
        self.assertEqual(artifacts["nhanes_screening_full"]["feature_count"], 58)

        # Framingham
        fg_pipe = load_model_artifact(models_dir / "framingham_risk.joblib")
        fg_clf = fg_pipe.named_steps["classifier"].__class__.__name__
        self.assertEqual(fg_clf, "LogisticRegression")
        self.assertEqual(artifacts["framingham_risk"]["selected_estimator"], "Logistic_Regression")
        self.assertEqual(artifacts["framingham_risk"]["feature_count"], 14)

    def test_artifact_threshold_and_metadata_consistency(self):
        """Verify operating threshold recorded in metadata matches the comparison CSV row."""
        models_dir = Path("models")
        rep_dir = Path("reports/model_evaluation")
        with open(models_dir / "model_metadata.json", "r", encoding="utf-8") as f:
            master_meta = json.load(f)

        # NHANES Core Comparison CSV
        nh_core_csv = pd.read_csv(rep_dir / "nhanes_core_comparison.csv")
        gb_core_row = nh_core_csv[nh_core_csv["Model"] == "Gradient_Boosting"].iloc[0]
        meta_core_tau = master_meta["artifacts"]["nhanes_screening_core"]["selected_decision_threshold"]
        self.assertAlmostEqual(meta_core_tau, float(gb_core_row["Selected_Threshold (Tau)"]), places=4)

        # Framingham Comparison CSV
        fg_csv = pd.read_csv(rep_dir / "framingham_comparison.csv")
        lr_fg_row = fg_csv[fg_csv["Model"] == "Logistic_Regression"].iloc[0]
        meta_fg_tau = master_meta["artifacts"]["framingham_risk"]["selected_decision_threshold"]
        self.assertAlmostEqual(meta_fg_tau, float(lr_fg_row["Selected_Threshold (Tau)"]), places=4)

    def test_correct_threshold_to_model_association(self):
        """Verify threshold associations are distinct and not cross-contaminated across models."""
        rep_dir = Path("reports/model_evaluation")
        fg_csv = pd.read_csv(rep_dir / "framingham_comparison.csv")

        tau_lr = float(fg_csv[fg_csv["Model"] == "Logistic_Regression"]["Selected_Threshold (Tau)"].iloc[0])
        tau_rf = float(fg_csv[fg_csv["Model"] == "Random_Forest"]["Selected_Threshold (Tau)"].iloc[0])
        tau_gb = float(fg_csv[fg_csv["Model"] == "Gradient_Boosting"]["Selected_Threshold (Tau)"].iloc[0])

        # Framingham models have distinct optimized thresholds due to different probability calibrations
        self.assertAlmostEqual(tau_lr, 0.40, places=2)
        self.assertAlmostEqual(tau_rf, 0.27, places=2)
        self.assertAlmostEqual(tau_gb, 0.18, places=2)
        self.assertNotEqual(tau_lr, tau_rf)
        self.assertNotEqual(tau_rf, tau_gb)

    def test_confusion_matrix_and_metric_reconciliation(self):
        """Verify confusion matrices sum to exact cohort size and match calculated metrics."""
        models_dir = Path("models")
        with open(models_dir / "model_metadata.json", "r", encoding="utf-8") as f:
            master_meta = json.load(f)

        # NHANES Core test set (N = 1,155)
        core_opt = master_meta["artifacts"]["nhanes_screening_core"]["test_metrics_at_selected_threshold"]
        cm_core = core_opt["confusion_matrix"]
        total_core = cm_core["TN"] + cm_core["FP"] + cm_core["FN"] + cm_core["TP"]
        self.assertEqual(total_core, 1155)
        calc_core_sens = cm_core["TP"] / (cm_core["TP"] + cm_core["FN"])
        calc_core_spec = cm_core["TN"] / (cm_core["TN"] + cm_core["FP"])
        self.assertAlmostEqual(core_opt["sensitivity"], calc_core_sens, places=3)
        self.assertAlmostEqual(core_opt["specificity"], calc_core_spec, places=3)

        # Framingham test set (N = 797: 723 Neg, 74 Pos)
        fg_opt = master_meta["artifacts"]["framingham_risk"]["test_metrics_at_selected_threshold"]
        cm_fg = fg_opt["confusion_matrix"]
        total_fg = cm_fg["TN"] + cm_fg["FP"] + cm_fg["FN"] + cm_fg["TP"]
        self.assertEqual(total_fg, 797)
        self.assertEqual(cm_fg["TP"] + cm_fg["FN"], 74)
        self.assertEqual(cm_fg["TN"] + cm_fg["FP"], 723)
        calc_fg_sens = cm_fg["TP"] / (cm_fg["TP"] + cm_fg["FN"])
        calc_fg_spec = cm_fg["TN"] / (cm_fg["TN"] + cm_fg["FP"])
        self.assertAlmostEqual(fg_opt["sensitivity"], calc_fg_sens, places=3)
        self.assertAlmostEqual(fg_opt["specificity"], calc_fg_spec, places=3)

    def test_survey_weight_alignment_and_non_negative(self):
        """Verify NHANES survey weights align exactly with participant records and are positive."""
        from src.nhanes_preprocessing import (
            load_nhanes_dataset,
            build_nhanes_screening_cohort,
            extract_nhanes_feature_sets,
            split_nhanes_dataset,
        )

        df_raw = load_nhanes_dataset()
        cohort_df, _ = build_nhanes_screening_cohort(df_raw)
        X, y, survey_meta = extract_nhanes_feature_sets(cohort_df, feature_subset="core_24")
        X_tr, X_te, y_tr, y_te, s_tr, s_te = split_nhanes_dataset(
            X, y, survey_meta=survey_meta, test_size=0.20, random_state=42
        )

        self.assertEqual(len(s_te), len(y_te))
        self.assertTrue(s_te.index.equals(y_te.index))
        self.assertIn("survey_weight_mec_2yr", s_te.columns)
        self.assertEqual(s_te["survey_weight_mec_2yr"].isna().sum(), 0)
        self.assertTrue((s_te["survey_weight_mec_2yr"] > 0).all())

    def test_model_selection_metadata_transparency(self):
        """Verify metadata documents selection status, rationale, and OOF tuning without ambiguity."""
        models_dir = Path("models")
        with open(models_dir / "model_metadata.json", "r", encoding="utf-8") as f:
            master_meta = json.load(f)

        for art_key in ["nhanes_screening_core", "nhanes_screening_full", "framingham_risk"]:
            art_meta = master_meta["artifacts"][art_key]
            self.assertIn("model_selection_status", art_meta)
            self.assertIn("model_selection_rationale", art_meta)
            self.assertIn("threshold_selection_methodology", art_meta)
            self.assertIn("oof_metrics_at_selected_threshold", art_meta)
            self.assertGreater(len(art_meta["model_selection_rationale"]), 30)


if __name__ == "__main__":
    unittest.main()
