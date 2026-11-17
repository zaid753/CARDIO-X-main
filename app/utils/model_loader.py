"""
CARDIO-X: Model Loading and Health Verification Utility.

Provides cached loading and validation for trained machine learning pipelines
and application metadata. Guarantees zero model retraining during runtime
and protects raw training datasets from unnecessary loading.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
from sklearn.pipeline import Pipeline
import streamlit as st

logger = logging.getLogger(__name__)

# Canonical model identifiers
ARTIFACT_KEYS = [
    "nhanes_screening_core",
    "nhanes_screening_full",
    "framingham_risk",
]

ARTIFACT_FILENAMES = {
    "nhanes_screening_core": "nhanes_screening_core.joblib",
    "nhanes_screening_full": "nhanes_screening_full.joblib",
    "framingham_risk": "framingham_risk.joblib",
}

EXPECTED_ESTIMATORS = {
    "nhanes_screening_core": "HistGradientBoostingClassifier",
    "nhanes_screening_full": "HistGradientBoostingClassifier",
    "framingham_risk": "LogisticRegression",
}

EXPECTED_THRESHOLDS = {
    "nhanes_screening_core": 0.46,
    "nhanes_screening_full": 0.46,
    "framingham_risk": 0.40,
}


def get_models_dir(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolves the models directory safely whether running from repo root or subdirectories.
    """
    if custom_path is not None:
        return Path(custom_path)

    # 1. Check relative to current working directory (standard streamlit run streamlit_app.py from repo root)
    cwd_models = Path("models").resolve()
    if cwd_models.is_dir():
        return cwd_models

    # 2. Check relative to this file: app/utils/model_loader.py -> repo_root/models
    file_models = Path(__file__).resolve().parent.parent.parent / "models"
    if file_models.is_dir():
        return file_models

    return cwd_models


@st.cache_data(show_spinner=False)
def load_model_metadata(models_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Loads and caches the consolidated model metadata registry.

    Returns
    -------
    Dict[str, Any]
        Dictionary with master metadata, or empty dict with error key on failure.
    """
    target_dir = get_models_dir(models_dir)
    metadata_path = target_dir / "model_metadata.json"

    if not metadata_path.is_file():
        logger.warning(f"Model metadata file missing: {metadata_path.name}")
        return {"error": f"Metadata file not found: {metadata_path.name}", "artifacts": {}}

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        logger.error(f"Failed to read model metadata: {exc}")
        return {"error": f"Corrupted metadata: {exc}", "artifacts": {}}


@st.cache_resource(show_spinner=False)
def load_pipeline(artifact_key: str, models_dir: Optional[Union[str, Path]] = None) -> Optional[Pipeline]:
    """
    Loads and caches a single scikit-learn Pipeline by artifact key.

    Parameters
    ----------
    artifact_key : str
        One of 'nhanes_screening_core', 'nhanes_screening_full', 'framingham_risk'.
    models_dir : Optional[Union[str, Path]]
        Optional explicit models directory path.

    Returns
    -------
    Optional[Pipeline]
        Fitted scikit-learn Pipeline, or None if loading failed.
    """
    if artifact_key not in ARTIFACT_FILENAMES:
        raise ValueError(f"Unknown artifact key '{artifact_key}'. Must be one of {list(ARTIFACT_FILENAMES.keys())}")

    target_dir = get_models_dir(models_dir)
    filename = ARTIFACT_FILENAMES[artifact_key]
    artifact_path = target_dir / filename

    if not artifact_path.is_file():
        logger.warning(f"Model artifact file not found: {filename}")
        return None

    try:
        pipeline = joblib.load(artifact_path)
        if not isinstance(pipeline, Pipeline):
            logger.error(f"Artifact {filename} is not a valid scikit-learn Pipeline instance.")
            return None
        return pipeline
    except Exception as exc:
        logger.error(f"Error deserializing {filename}: {exc}")
        return None


def verify_models_health(models_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Performs comprehensive runtime health verification across all model artifacts:
    - Verifies file presence on disk.
    - Confirms deserialization without retraining.
    - Verifies estimator class matches expected architecture.
    - Verifies metadata presence, matching estimator, and matching decision threshold.

    Returns
    -------
    Dict[str, Any]
        Structured health report containing overall status and per-model diagnostics.
    """
    metadata = load_model_metadata(models_dir=models_dir)
    has_meta_error = "error" in metadata and bool(metadata["error"])
    artifacts_meta = metadata.get("artifacts", {})

    report: Dict[str, Any] = {
        "overall_status": "healthy",
        "metadata_loaded": not has_meta_error and bool(artifacts_meta),
        "models": {},
    }

    for key in ARTIFACT_KEYS:
        expected_est = EXPECTED_ESTIMATORS[key]
        expected_tau = EXPECTED_THRESHOLDS[key]
        model_meta = artifacts_meta.get(key, {})

        pipeline = load_pipeline(key, models_dir=models_dir)
        is_loaded = pipeline is not None

        est_class = None
        est_match = False
        if is_loaded and hasattr(pipeline, "named_steps"):
            classifier = pipeline.named_steps.get("classifier")
            if classifier is not None:
                est_class = classifier.__class__.__name__
                est_match = (est_class == expected_est)

        meta_tau = model_meta.get("selected_decision_threshold")
        tau_match = False
        if meta_tau is not None:
            tau_match = abs(float(meta_tau) - expected_tau) < 0.02

        # Status categorization
        if is_loaded and est_match and tau_match:
            status = "operational"
        elif is_loaded and est_match:
            status = "degraded_threshold_mismatch"
            report["overall_status"] = "degraded"
        elif is_loaded:
            status = "degraded_estimator_mismatch"
            report["overall_status"] = "degraded"
        else:
            status = "missing"
            report["overall_status"] = "degraded"

        report["models"][key] = {
            "status": status,
            "loaded": is_loaded,
            "estimator_class": est_class,
            "expected_estimator": expected_est,
            "estimator_matches": est_match,
            "operating_threshold": meta_tau if meta_tau is not None else expected_tau,
            "expected_threshold": expected_tau,
            "threshold_matches": tau_match,
            "feature_count": model_meta.get("feature_count", len(model_meta.get("feature_names", []))),
            "model_tier": model_meta.get("model_tier", "Unknown Tier"),
            "target_name": model_meta.get("target_name", "Unknown Target"),
            "selection_status": model_meta.get("model_selection_status", "unverified"),
        }

    return report
