"""
Unit and integration tests for CARDIO-X Streamlit Application Foundation (Phase 3A).
Tests model loading, metadata caching, graceful failure on missing artifacts,
component imports, zero retraining, and Streamlit execution readiness.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.utils.model_loader import (
    load_model_metadata,
    load_pipeline,
    verify_models_health,
    get_models_dir,
    ARTIFACT_KEYS,
    EXPECTED_ESTIMATORS,
    EXPECTED_THRESHOLDS,
)


class TestStreamlitAppFoundation(unittest.TestCase):
    """Test suite for Phase 3A Streamlit application foundation."""

    def test_model_loader_metadata_loading(self):
        """Verify metadata loads correctly and contains valid artifacts."""
        meta = load_model_metadata()
        self.assertNotIn("error", meta)
        self.assertIn("artifacts", meta)
        for key in ARTIFACT_KEYS:
            self.assertIn(key, meta["artifacts"])
            art = meta["artifacts"][key]
            self.assertIn("selected_estimator", art)
            self.assertIn("selected_decision_threshold", art)
            self.assertIn("feature_count", art)

    def test_model_loader_pipeline_loading(self):
        """Verify all three saved pipelines load and are valid scikit-learn Pipelines."""
        for key in ARTIFACT_KEYS:
            pipe = load_pipeline(key)
            self.assertIsNotNone(pipe, f"Pipeline {key} failed to load.")
            self.assertIsInstance(pipe, Pipeline)
            self.assertTrue(hasattr(pipe, "predict_proba"))
            self.assertIn("preprocessor", pipe.named_steps)
            self.assertIn("classifier", pipe.named_steps)

    def test_model_loader_unknown_key_rejection(self):
        """Verify attempting to load an unregistered artifact key raises ValueError."""
        with self.assertRaises(ValueError):
            load_pipeline("unregistered_engine_xyz")

    def test_missing_artifact_graceful_handling(self):
        """Verify loader handles missing files cleanly without crashing."""
        with tempfile.TemporaryDirectory() as empty_dir:
            # Metadata missing
            meta = load_model_metadata(models_dir=empty_dir)
            self.assertIn("error", meta)

            # Pipeline missing
            pipe = load_pipeline("nhanes_screening_core", models_dir=empty_dir)
            self.assertIsNone(pipe)

            # Health check with missing files
            health = verify_models_health(models_dir=empty_dir)
            self.assertEqual(health["overall_status"], "degraded")
            self.assertFalse(health["metadata_loaded"])
            for key in ARTIFACT_KEYS:
                self.assertEqual(health["models"][key]["status"], "missing")
                self.assertFalse(health["models"][key]["loaded"])

    def test_verify_models_health_operational_on_actual_artifacts(self):
        """Verify runtime health check passes completely on real project artifacts."""
        health = verify_models_health()
        self.assertEqual(health["overall_status"], "healthy")
        self.assertTrue(health["metadata_loaded"])
        for key in ARTIFACT_KEYS:
            model_info = health["models"][key]
            self.assertEqual(model_info["status"], "operational")
            self.assertTrue(model_info["loaded"])
            self.assertTrue(model_info["estimator_matches"])
            self.assertTrue(model_info["threshold_matches"])
            self.assertEqual(model_info["estimator_class"], EXPECTED_ESTIMATORS[key])

    def test_no_model_retraining_during_app_loading(self):
        """Verify loading pipelines does not mutate or rewrite model files."""
        models_dir = get_models_dir()
        mtimes_before = {}
        for key in ARTIFACT_KEYS:
            art_file = models_dir / f"{key}.joblib"
            if art_file.is_file():
                mtimes_before[key] = art_file.stat().st_mtime_ns

        # Perform loading operations
        _ = verify_models_health()
        for key in ARTIFACT_KEYS:
            _ = load_pipeline(key)

        for key in ARTIFACT_KEYS:
            art_file = models_dir / f"{key}.joblib"
            if key in mtimes_before:
                self.assertEqual(
                    art_file.stat().st_mtime_ns,
                    mtimes_before[key],
                    f"File {art_file.name} was modified during loading (unintended retraining)!",
                )

    def test_all_app_modules_import_cleanly(self):
        """Verify all application packages, views, and components import without syntax or import errors."""
        import app.streamlit_app as st_app
        import app.components.header as header
        import app.components.sidebar as sidebar
        import app.components.status_cards as status_cards
        import app.pages.dashboard as dashboard
        import app.pages.nhanes_screening as nhanes
        import app.pages.framingham_risk as framingham
        import app.pages.lipid_analysis as lipid
        import app.pages.lifestyle as lifestyle
        import app.pages.reports_view as reports

        self.assertTrue(callable(st_app.main))
        self.assertTrue(callable(header.render_header))
        self.assertTrue(callable(sidebar.render_sidebar))
        self.assertTrue(callable(status_cards.render_model_status_card))
        self.assertTrue(callable(dashboard.render_dashboard_page))
        self.assertTrue(callable(nhanes.render_nhanes_page))
        self.assertTrue(callable(framingham.render_framingham_page))
        self.assertTrue(callable(lipid.render_lipid_page))
        self.assertTrue(callable(lifestyle.render_lifestyle_page))
        self.assertTrue(callable(reports.render_reports_page))

    def test_streamlit_startup_headless_smoke_test(self):
        """Verify app/streamlit_app.py runs without syntax or runtime exceptions in Python."""
        # Execute streamlit_app.py under Python directly with a mock/test check
        cmd = [
            sys.executable,
            "-c",
            "import app.streamlit_app; print('Streamlit app module imported successfully')",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_repo_root))
        self.assertEqual(result.returncode, 0, f"Import smoke test failed: {result.stderr}")
        self.assertIn("Streamlit app module imported successfully", result.stdout)


if __name__ == "__main__":
    unittest.main()
