"""
Cardio-X: NHANES Hypercholesterolemia Screening Preprocessing Module.

Provides reproducible, leakage-free preprocessing, cohort construction,
non-invasive predictor selection, and model-tailored Scikit-Learn transformers
for the NHANES 2021-2023 screening model (CARDIO-X Model 1).
Guarantees zero in-place mutation of the raw dataset.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# ===========================================================================
# 1. Audited Feature Registry and Exclusion Taxonomy (from ml_feature_dictionary.csv)
# ===========================================================================

# 58 Audited Non-Invasive Candidate Predictors
AUDITED_NON_INVASIVE_PREDICTORS: List[str] = [
    # Demographics (6)
    "age_years",
    "sex",
    "race_ethnicity",
    "education_level",
    "marital_status",
    "born_in_us",
    # Socioeconomic (2)
    "family_income_poverty_ratio",
    "income_poverty_category",
    # Pregnancy & Employment (2)
    "pregnant",
    "work_status",
    # Anthropometrics & Body Composition (10)
    "bmi_kg_m2",
    "weight_kg",
    "height_cm",
    "waist_circumference_cm",
    "hip_circumference_cm",
    "arm_circumference_cm",
    "waist_to_height_ratio",
    "waist_to_hip_ratio",
    "weight_change_1yr_kg",
    "tried_to_lose_weight_past_year",
    # Vitals & Hemodynamics (7)
    "avg_systolic_bp_mmhg",
    "avg_diastolic_bp_mmhg",
    "avg_pulse_bpm",
    "pulse_pressure_mmhg",
    "hypertension_measured",
    "hypertension_diagnosed",
    "bp_medication",
    # Physical Activity (4)
    "moderate_ltp_minutes_week",
    "vigorous_ltp_minutes_week",
    "total_ltp_met_minutes_week",
    "meets_activity_guidelines",
    # Sedentary & Sleep (3)
    "sedentary_minutes_day",
    "sleep_hours_weekday",
    "sleep_hours_weekend",
    # Tobacco & Alcohol (4)
    "smoked_100_cigarettes",
    "smoking_status",
    "alcohol_drinks_per_week",
    "heavy_alcohol_use",
    # Mental Health & Health Status (4)
    "phq9_score",
    "depression_screen_positive",
    "general_health",
    "general_health_label",
    # Healthcare Utilization (3)
    "has_health_insurance",
    "routine_healthcare_place",
    "mental_health_care_past_year",
    # Medical History (13)
    "asthma",
    "congestive_heart_failure",
    "coronary_heart_disease",
    "angina",
    "heart_attack",
    "stroke",
    "liver_condition",
    "thyroid_problem",
    "cancer_any",
    "weak_or_failing_kidneys",
    "heart_disease_or_attack",
    "diabetes",
    "diabetes_diagnosed",
]

# 24 Core Curated Non-Invasive Predictors (Section 3.C of audit report)
RECOMMENDED_CORE_PREDICTORS: List[str] = [
    # Demographics (5)
    "age_years",
    "sex",
    "race_ethnicity",
    "education_level",
    "family_income_poverty_ratio",
    # Anthropometrics (4)
    "bmi_kg_m2",
    "waist_circumference_cm",
    "waist_to_height_ratio",
    "weight_change_1yr_kg",
    # Vitals & Hemodynamics (5)
    "avg_systolic_bp_mmhg",
    "avg_diastolic_bp_mmhg",
    "pulse_pressure_mmhg",
    "avg_pulse_bpm",
    "bp_medication",
    # Lifestyle & Habits (5)
    "smoking_status",
    "total_ltp_met_minutes_week",
    "sedentary_minutes_day",
    "alcohol_drinks_per_week",
    "sleep_hours_weekday",
    # Medical History (5)
    "diabetes_diagnosed",
    "general_health",
    "phq9_score",
    "coronary_heart_disease",
    "stroke",
]

# Excluded Variables (Strictly prohibited from predictor matrices)
EXCLUDED_VARIABLES: List[str] = [
    # Direct Target Definers & Lipid Laboratory Measurements (8)
    "total_cholesterol_mg_dl",
    "cholesterol_medication",
    "high_cholesterol_diagnosed",
    "non_hdl_cholesterol_mg_dl",
    "ldl_cholesterol_mg_dl",
    "triglycerides_mg_dl",
    "triglyceride_hdl_ratio",
    "hdl_cholesterol_mg_dl",
    # Hidden Laboratory Contamination (2)
    "diabetes_undiagnosed",
    "glycemic_status",
    # Invasive Laboratory Tests (10)
    "alt_u_l",
    "ast_u_l",
    "ggt_u_l",
    "uric_acid_mg_dl",
    "creatinine_mg_dl",
    "albumin_g_dl",
    "egfr_ml_min_1_73m2",
    "white_blood_cell_count_1000_ul",
    "hemoglobin_g_dl",
    "platelet_count_1000_ul",
    # Survey Design & Sampling Identifiers (7)
    "participant_id",
    "survey_cycle",
    "survey_weight_interview_2yr",
    "survey_weight_mec_2yr",
    "survey_weight_fasting_2yr",
    "survey_strata",
    "survey_psu",
    # Redundant Collinear Binned Variables (3)
    "age_group",
    "education_code",
    "bmi_category",
]

# Survey Design Metadata Columns (preserved separately for survey-weighted evaluation)
SURVEY_METADATA_COLUMNS: List[str] = [
    "participant_id",
    "survey_cycle",
    "survey_weight_interview_2yr",
    "survey_weight_mec_2yr",
    "survey_weight_fasting_2yr",
    "survey_strata",
    "survey_psu",
]

# Feature Data Type Partitioning within the 58 Audited Predictors
CONTINUOUS_PREDICTORS: List[str] = [
    "age_years",
    "family_income_poverty_ratio",
    "bmi_kg_m2",
    "weight_kg",
    "height_cm",
    "waist_circumference_cm",
    "hip_circumference_cm",
    "arm_circumference_cm",
    "waist_to_height_ratio",
    "waist_to_hip_ratio",
    "weight_change_1yr_kg",
    "avg_systolic_bp_mmhg",
    "avg_diastolic_bp_mmhg",
    "avg_pulse_bpm",
    "pulse_pressure_mmhg",
    "moderate_ltp_minutes_week",
    "vigorous_ltp_minutes_week",
    "total_ltp_met_minutes_week",
    "sedentary_minutes_day",
    "alcohol_drinks_per_week",
    "sleep_hours_weekday",
    "sleep_hours_weekend",
    "phq9_score",
]

BINARY_ORDINAL_PREDICTORS: List[str] = [
    "born_in_us",
    "income_poverty_category",
    "pregnant",
    "tried_to_lose_weight_past_year",
    "hypertension_measured",
    "hypertension_diagnosed",
    "bp_medication",
    "meets_activity_guidelines",
    "smoked_100_cigarettes",
    "heavy_alcohol_use",
    "depression_screen_positive",
    "general_health",
    "has_health_insurance",
    "routine_healthcare_place",
    "mental_health_care_past_year",
    "asthma",
    "congestive_heart_failure",
    "coronary_heart_disease",
    "angina",
    "heart_attack",
    "stroke",
    "liver_condition",
    "thyroid_problem",
    "cancer_any",
    "weak_or_failing_kidneys",
    "heart_disease_or_attack",
    "diabetes",
    "diabetes_diagnosed",
]

CATEGORICAL_PREDICTORS: List[str] = [
    "sex",
    "race_ethnicity",
    "education_level",
    "marital_status",
    "work_status",
    "smoking_status",
    "general_health_label",
]


# ===========================================================================
# 2. Pipeline Functions
# ===========================================================================

def load_nhanes_dataset(
    filepath: str = "data_comparison/nhanes_diabetes_2021_2023.csv"
) -> pd.DataFrame:
    """
    Safely loads the raw NHANES microdata dataset without mutating source file.

    Parameters
    ----------
    filepath : str
        Path to the NHANES CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded pandas DataFrame.
    """
    in_file = Path(filepath)
    if not in_file.is_file():
        raise FileNotFoundError(f"NHANES dataset not found at expected path: {filepath}")

    return pd.read_csv(in_file)


def validate_nhanes_columns(df: pd.DataFrame) -> bool:
    """
    Validates presence of essential target component columns and expected predictors.

    Parameters
    ----------
    df : pd.DataFrame
        NHANES DataFrame to validate.

    Returns
    -------
    bool
        True if all required columns exist; raises ValueError otherwise.
    """
    required_target_cols = ["total_cholesterol_mg_dl", "cholesterol_medication"]
    for col in required_target_cols:
        if col not in df.columns:
            raise ValueError(f"Required target component column '{col}' missing from NHANES dataset.")

    missing_predictors = [p for p in AUDITED_NON_INVASIVE_PREDICTORS if p not in df.columns]
    if missing_predictors:
        raise ValueError(
            f"Expected {len(missing_predictors)} predictor(s) missing from dataset: {missing_predictors[:5]}..."
        )

    return True


def build_nhanes_screening_cohort(
    df: pd.DataFrame, return_audit: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Constructs the evaluable screening cohort using audited Target Option B:

    TARGET_HYPERCHOL = 1 if:
        - total_cholesterol_mg_dl >= 240
        OR
        - cholesterol_medication == 1

    Otherwise TARGET_HYPERCHOL = 0, provided both components are sufficiently
    observed to rule out hypercholesterolemia:
        - cholesterol_medication == 0 AND total_cholesterol_mg_dl < 240

    Indeterminate cases (unobserved target status) are excluded and audited:
        - cholesterol_medication == 0 and total_cholesterol_mg_dl is NaN (N=202)
        - cholesterol_medication is NaN and total_cholesterol_mg_dl < 240 (N=23)
        - Both components are NaN (N=2)

    Parameters
    ----------
    df : pd.DataFrame
        Raw NHANES DataFrame.
    return_audit : bool
        If True, returns step-by-step attrition metrics.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        The evaluable cohort with TARGET_HYPERCHOL, and the audit dictionary.
    """
    validate_nhanes_columns(df)

    audit: Dict[str, Any] = {}
    total_raw = len(df)
    audit["total_raw_rows"] = total_raw

    tc = df["total_cholesterol_mg_dl"]
    med = df["cholesterol_medication"]

    # 1. Definite Positive Condition: Medicated OR measured >= 240 mg/dL
    pos_mask = (med == 1.0) | (tc >= 240.0)

    # 2. Definite Negative Condition: Unmedicated AND measured < 240 mg/dL
    neg_mask = (med == 0.0) & (tc < 240.0)

    # 3. Indeterminate Condition: Neither positive nor definitively negative
    indet_mask = ~(pos_mask | neg_mask)

    # Detailed indeterminate breakdown
    indet_tc_null_med_zero = (tc.isnull() & (med == 0.0)).sum()
    indet_med_null_tc_low = (med.isnull() & (tc < 240.0)).sum()
    indet_both_null = (tc.isnull() & med.isnull()).sum()

    n_positives = int(pos_mask.sum())
    n_negatives = int(neg_mask.sum())
    n_indeterminate = int(indet_mask.sum())

    audit["positive_cases"] = n_positives
    audit["negative_cases"] = n_negatives
    audit["indeterminate_cases"] = n_indeterminate
    audit["indeterminate_breakdown"] = {
        "tc_null_unmedicated": int(indet_tc_null_med_zero),
        "med_null_tc_normal": int(indet_med_null_tc_low),
        "both_null": int(indet_both_null),
    }

    # Filter out indeterminate rows
    evaluable_df = df[pos_mask | neg_mask].copy()
    evaluable_df["TARGET_HYPERCHOL"] = np.where(pos_mask[pos_mask | neg_mask], 1, 0)

    n_evaluable = len(evaluable_df)
    audit["evaluable_rows"] = n_evaluable
    audit["prevalence_pct"] = round(n_positives / n_evaluable * 100.0, 2) if n_evaluable > 0 else 0.0

    return evaluable_df, audit


def extract_nhanes_feature_sets(
    df: pd.DataFrame, feature_subset: str = "all_58"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extracts strictly leak-free predictor matrix X, binary target y, and survey metadata.
    Guarantees that zero target definers, proxies, or invasive labs enter X.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluable NHANES cohort containing TARGET_HYPERCHOL.
    feature_subset : str
        'all_58' (default: full audited non-invasive registry) or
        'core_24' (curated high-impact clinical/lifestyle subset).

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        (X, y, survey_metadata)
    """
    if "TARGET_HYPERCHOL" not in df.columns:
        raise ValueError("TARGET_HYPERCHOL missing from dataframe. Call build_nhanes_screening_cohort() first.")

    # Select feature columns
    if feature_subset == "all_58":
        feature_cols = [c for c in AUDITED_NON_INVASIVE_PREDICTORS if c in df.columns]
    elif feature_subset == "core_24":
        feature_cols = [c for c in RECOMMENDED_CORE_PREDICTORS if c in df.columns]
    else:
        raise ValueError(f"Unknown feature_subset '{feature_subset}'. Must be 'all_58' or 'core_24'.")

    # Strict Zero-Leakage Assertion
    for col in feature_cols:
        if col in EXCLUDED_VARIABLES:
            raise ValueError(f"CRITICAL TARGET LEAKAGE: Excluded variable '{col}' found in feature set.")
        if "cholesterol" in col.lower() or "ldl" in col.lower() or "trig" in col.lower():
            raise ValueError(f"CRITICAL TARGET LEAKAGE: Lipid-related variable '{col}' found in feature set.")

    X = df[feature_cols].copy()
    y = df["TARGET_HYPERCHOL"].astype(int).copy()

    # Preserve survey design metadata separately
    present_survey_cols = [c for c in SURVEY_METADATA_COLUMNS if c in df.columns]
    survey_metadata = df[present_survey_cols].copy()

    return X, y, survey_metadata


def get_nhanes_feature_types(X: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classifies columns of feature matrix X into continuous, binary/ordinal, and categorical.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor matrix.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with keys 'continuous', 'binary_ordinal', and 'categorical'.
    """
    cols = list(X.columns)

    cont = [c for c in CONTINUOUS_PREDICTORS if c in cols]
    bin_ord = [c for c in BINARY_ORDINAL_PREDICTORS if c in cols]
    cat = [c for c in CATEGORICAL_PREDICTORS if c in cols]

    # Safety catch for any column not pre-categorized
    assigned = set(cont + bin_ord + cat)
    unassigned = [c for c in cols if c not in assigned]
    for c in unassigned:
        if X[c].dtype == "object" or pd.api.types.is_string_dtype(X[c]):
            cat.append(c)
        elif X[c].nunique(dropna=True) <= 2:
            bin_ord.append(c)
        else:
            cont.append(c)

    return {
        "continuous": cont,
        "binary_ordinal": bin_ord,
        "categorical": cat,
    }


def create_nhanes_preprocessor(
    feature_subset: str = "all_58",
    model_family: str = "tree"
) -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer preprocessor for NHANES screening.
    Fit parameters are calculated strictly on training folds.

    Rules:
    - Continuous features: SimpleImputer(strategy='median')
      - 'linear' models: followed by RobustScaler() to protect against skewed biometrics.
      - 'tree' models: no scaling (preserves natural units).
    - Binary / Ordinal features: SimpleImputer(strategy='most_frequent') with passthrough.
    - Categorical features: SimpleImputer(strategy='most_frequent') followed by
      OneHotEncoder(handle_unknown='ignore', sparse_output=False).

    Parameters
    ----------
    feature_subset : str
        'all_58' or 'core_24'.
    model_family : str
        'tree' or 'linear'.

    Returns
    -------
    ColumnTransformer
        Configured preprocessor instance.
    """
    if feature_subset == "all_58":
        target_features = AUDITED_NON_INVASIVE_PREDICTORS
    elif feature_subset == "core_24":
        target_features = RECOMMENDED_CORE_PREDICTORS
    else:
        raise ValueError(f"Unknown feature_subset '{feature_subset}'. Must be 'all_58' or 'core_24'.")

    cont_cols = [c for c in CONTINUOUS_PREDICTORS if c in target_features]
    bin_ord_cols = [c for c in BINARY_ORDINAL_PREDICTORS if c in target_features]
    cat_cols = [c for c in CATEGORICAL_PREDICTORS if c in target_features]

    # Continuous Pipeline
    if model_family == "linear":
        cont_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])
    else:
        # Tree-based: median imputation without scaling
        cont_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    # Binary / Ordinal Pipeline (mode imputation, passthrough)
    bin_ord_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    # Categorical Pipeline (mode imputation + one-hot encoding)
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if cont_cols:
        transformers.append(("cont", cont_pipeline, cont_cols))
    if bin_ord_cols:
        transformers.append(("bin_ord", bin_ord_pipeline, bin_ord_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def split_nhanes_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    survey_meta: Optional[pd.DataFrame] = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    """
    Executes a reproducible Stratified Train/Test split for NHANES.
    Stratifies on target y to ensure identical prevalence in both partitions.
    Maintains exact index alignment across X, y, and survey_meta.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor matrix.
    y : pd.Series
        Binary target vector (TARGET_HYPERCHOL).
    survey_meta : Optional[pd.DataFrame]
        Associated survey weights and strata.
    test_size : float
        Fraction reserved for held-out evaluation (default: 0.20).
    random_state : int
        Deterministic random seed (default: 42).

    Returns
    -------
    Tuple
        (X_train, X_test, y_train, y_test, survey_train, survey_test)
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    survey_train = survey_meta.iloc[train_idx].copy() if survey_meta is not None else None
    survey_test = survey_meta.iloc[test_idx].copy() if survey_meta is not None else None

    return X_train, X_test, y_train, y_test, survey_train, survey_test


def summarize_nhanes_preprocessing(
    audit: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> str:
    """
    Generates a structured text summary of the NHANES preprocessing pipeline.
    """
    lines = [
        "=" * 60,
        "CARDIO-X: NHANES SCREENING PREPROCESSING SUMMARY",
        "=" * 60,
        f"Raw NHANES Records     : {audit.get('total_raw_rows', 0):,}",
        f"Evaluable Cohort Size  : {audit.get('evaluable_rows', 0):,}",
        f"Indeterminate Excluded : {audit.get('indeterminate_cases', 0):,} "
        f"({audit.get('indeterminate_cases', 0)/audit.get('total_raw_rows', 1)*100:.2f}%)",
        f"  - TC null, unmedicated : {audit.get('indeterminate_breakdown', {}).get('tc_null_unmedicated', 0):,}",
        f"  - Med null, normal TC  : {audit.get('indeterminate_breakdown', {}).get('med_null_tc_normal', 0):,}",
        f"  - Both null            : {audit.get('indeterminate_breakdown', {}).get('both_null', 0):,}",
        f"Confirmed Positives (Y=1): {audit.get('positive_cases', 0):,} ({audit.get('prevalence_pct', 0.0):.2f}%)",
        f"Confirmed Negatives (Y=0): {audit.get('negative_cases', 0):,} ({100 - audit.get('prevalence_pct', 0.0):.2f}%)",
        "-" * 60,
        "PARTITIONING (Stratified 80/20, random_state=42):",
        f"Training Set (80%)     : {len(X_train):,} samples | Positives: {int(y_train.sum()):,} ({y_train.mean()*100:.2f}%)",
        f"Held-Out Test Set (20%): {len(X_test):,} samples | Positives: {int(y_test.sum()):,} ({y_test.mean()*100:.2f}%)",
        f"Feature Matrix Shape   : {X_train.shape[1]} input columns",
        "=" * 60,
    ]
    return "\n".join(lines)


def run_nhanes_preprocessing_pipeline(
    input_path: str = "data_comparison/nhanes_diabetes_2021_2023.csv",
    feature_subset: str = "all_58",
    test_size: float = 0.20,
    random_state: int = 42,
    assert_expected_counts: bool = True,
) -> Dict[str, Any]:
    """
    Executes the end-to-end NHANES preprocessing workflow:
    1. Cryptographically hashes the input file before execution.
    2. Loads and validates the dataset.
    3. Builds the Target Option B screening cohort.
    4. Extracts non-invasive feature matrices and survey design metadata.
    5. Performs stratified train/test splitting.
    6. Fits model preprocessors strictly on the training fold.
    7. Verifies input file immutability post-execution.

    Parameters
    ----------
    input_path : str
        Path to raw NHANES dataset.
    feature_subset : str
        'all_58' or 'core_24'.
    test_size : float
        Fraction for held-out test split.
    random_state : int
        Deterministic seed.
    assert_expected_counts : bool
        Asserts exact audit counts when run on the primary dataset.

    Returns
    -------
    Dict[str, Any]
        Dictionary with processed arrays, preprocessors, and audit metadata.
    """
    in_file = Path(input_path)
    if not in_file.is_file():
        raise FileNotFoundError(f"Source NHANES dataset not found at: {input_path}")

    # 1. Pre-execution cryptographic SHA-256 digest
    with open(in_file, "rb") as f:
        pre_hash = hashlib.sha256(f.read()).hexdigest()

    # 2. Load dataset
    df_raw = pd.read_csv(in_file)

    # 3. Build cohort with Target Option B
    eval_df, audit = build_nhanes_screening_cohort(df_raw, return_audit=True)

    # Assert expected counts if auditing full primary dataset
    if assert_expected_counts and audit["total_raw_rows"] == 6001:
        assert audit["total_raw_rows"] == 6001, f"Raw rows mismatch: {audit['total_raw_rows']}"
        assert audit["positive_cases"] == 2196, f"Positive cases mismatch: {audit['positive_cases']}"
        assert audit["negative_cases"] == 3578, f"Negative cases mismatch: {audit['negative_cases']}"
        assert audit["indeterminate_cases"] == 227, f"Indeterminate mismatch: {audit['indeterminate_cases']}"
        assert audit["evaluable_rows"] == 5774, f"Evaluable rows mismatch: {audit['evaluable_rows']}"
        assert audit["prevalence_pct"] == 38.03, f"Prevalence mismatch: {audit['prevalence_pct']}"

    # 4. Extract feature matrices and survey metadata
    X, y, survey_meta = extract_nhanes_feature_sets(eval_df, feature_subset=feature_subset)

    # 5. Stratified Train/Test split
    X_train, X_test, y_train, y_test, survey_train, survey_test = split_nhanes_dataset(
        X, y, survey_meta=survey_meta, test_size=test_size, random_state=random_state
    )

    if assert_expected_counts and len(eval_df) == 5774 and test_size == 0.20:
        assert len(X_train) == 4619, f"Train count mismatch: {len(X_train)}"
        assert len(X_test) == 1155, f"Test count mismatch: {len(X_test)}"
        assert int(y_train.sum()) == 1757, f"Train positive mismatch: {int(y_train.sum())}"
        assert int(y_test.sum()) == 439, f"Test positive mismatch: {int(y_test.sum())}"

    # 6. Fit transformers strictly on training fold
    pipe_linear = create_nhanes_preprocessor(feature_subset=feature_subset, model_family="linear")
    X_train_linear = pipe_linear.fit_transform(X_train)
    X_test_linear = pipe_linear.transform(X_test)

    pipe_tree = create_nhanes_preprocessor(feature_subset=feature_subset, model_family="tree")
    X_train_tree = pipe_tree.fit_transform(X_train)
    X_test_tree = pipe_tree.transform(X_test)

    # Verify no NaNs exist post-transformation
    assert not np.isnan(X_train_linear).any(), "NaN found in transformed linear training matrix."
    assert not np.isnan(X_test_linear).any(), "NaN found in transformed linear test matrix."
    assert not np.isnan(X_train_tree).any(), "NaN found in transformed tree training matrix."
    assert not np.isnan(X_test_tree).any(), "NaN found in transformed tree test matrix."

    # 7. Post-execution cryptographic SHA-256 verification
    with open(in_file, "rb") as f:
        post_hash = hashlib.sha256(f.read()).hexdigest()

    if pre_hash != post_hash:
        raise RuntimeError("Fatal: The raw NHANES dataset was modified during preprocessing execution!")

    summary_text = summarize_nhanes_preprocessing(audit, X_train, X_test, y_train, y_test)

    return {
        "status": "success",
        "input_hash": post_hash,
        "input_file_unchanged": True,
        "audit": audit,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count_raw": X.shape[1],
        "feature_count_transformed": X_train_linear.shape[1],
        "linear_preprocessor": pipe_linear,
        "tree_preprocessor": pipe_tree,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "survey_train": survey_train,
        "survey_test": survey_test,
        "X_train_linear": X_train_linear,
        "X_test_linear": X_test_linear,
        "X_train_tree": X_train_tree,
        "X_test_tree": X_test_tree,
        "summary": summary_text,
    }


if __name__ == "__main__":
    result = run_nhanes_preprocessing_pipeline()
    print(result["summary"])
