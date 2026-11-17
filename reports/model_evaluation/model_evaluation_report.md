# CARDIO-X: Phase 2B Machine Learning Model Benchmark & Evaluation Report

> **Generated:** 2026-11-17 | **Repository Root:** `/Users/zaid/Desktop/CARDIO-X-main`

## 1. Executive Summary
This report details the cross-validated and held-out test set performance of the two independent machine learning engines comprising CARDIO-X:
- **Model 1 (NHANES Screening Engine):** Non-laboratory screening model predicting phenotypic hypercholesterolemia (`TARGET_HYPERCHOL` under Option B: total cholesterol >= 240 mg/dL or cholesterol-lowering medication). Evaluated across Core 24 non-laboratory predictors (demographics, anthropometrics, vitals/hemodynamics, lifestyle habits, and selected medical history) and Full 58 non-invasive candidate features.
- **Model 2 (Framingham Risk Engine):** Conditional landmark binary classification predicting first incident hard cardiovascular disease within 10 years (`TARGET_10Y_CVD`) in the primary prevention cohort (N = 3,983, 9.26% 10-year event rate). Not a competing-risk survival model.

---

## 2. NHANES Experiment A: Core 24-Feature Configuration
Predictor Battery: 24 non-laboratory features including demographics (5), anthropometrics (4), vitals/hemodynamics (5), lifestyle (5), and selected medical history (diagnosed diabetes, general health score, PHQ-9 depression score, coronary heart disease history, and stroke history).

| Model | CV_ROC_AUC_Mean | CV_ROC_AUC_Std | CV_PR_AUC_Mean | CV_PR_AUC_Std | CV_Brier_Mean | Test_ROC_AUC | Test_PR_AUC | Test_Brier | Test_Acc_Def (0.50) | Test_Sens_Def (0.50) | Test_Spec_Def (0.50) | Selected_Threshold (Tau) | Test_Acc_Opt | Test_Sens_Opt | Test_Spec_Opt | Test_F1_Opt | Survey_Weighted_ROC_AUC | Survey_Weighted_Sens_Opt | Survey_Weighted_Spec_Opt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic_Regression | 0.8122 | 0.0084 | 0.7061 | 0.0182 | 0.1778 | 0.8349 | 0.736 | 0.1652 | 0.7515 | 0.7449 | 0.7556 | 0.45000000000000007 | 0.7403 | 0.7973 | 0.7053 | 0.7 | 0.8402 | 0.6871 | 0.7945 |
| Random_Forest | 0.8137 | 0.0082 | 0.7074 | 0.0138 | 0.1761 | 0.8289 | 0.7284 | 0.1672 | 0.7316 | 0.7563 | 0.7165 | 0.4800000000000001 | 0.7385 | 0.7995 | 0.7011 | 0.6992 | 0.8322 | 0.7161 | 0.7825 |
| Gradient_Boosting | 0.811 | 0.0059 | 0.7023 | 0.0135 | 0.1765 | 0.8205 | 0.7159 | 0.1701 | 0.7368 | 0.754 | 0.7263 | 0.4600000000000001 | 0.7212 | 0.779 | 0.6858 | 0.6799 | 0.8242 | 0.6761 | 0.7669 |

### Key Findings & Model Selection Transparency:
- **Cross-Validation Stability:** Across 5 stratified folds on the training set (N=4,619), all three models demonstrated comparable discrimination within one standard deviation: Random Forest (ROC-AUC: 0.8137 +/- 0.0082), Logistic Regression (0.8122 +/- 0.0084), and Gradient Boosting (0.8110 +/- 0.0059).
- **Threshold Optimization:** Deriving thresholds strictly from training-set out-of-fold (OOF) predictions targeting >= 80% sensitivity selected tau = 0.45 for Logistic Regression, tau = 0.48 for Random Forest, and tau = 0.46 for Gradient Boosting. All models satisfied the sensitivity constraint in OOF without fallback.
- **OOF vs. Unseen Test Generalization:** Achieving >= 80% sensitivity on training OOF predictions does not guarantee >= 80% sensitivity on unseen test data. On held-out test data (N=1,155), Logistic Regression achieved 79.73% sensitivity / 70.53% specificity; Random Forest achieved 79.95% sensitivity / 70.11% specificity; Gradient Boosting achieved 77.90% sensitivity / 68.58% specificity.
- **Test Discrimination:** On held-out test cases, Logistic Regression achieved the highest discriminative power (Test ROC-AUC: 0.8349, PR-AUC: 0.7360, Brier: 0.1652), followed by Random Forest (0.8289, PR-AUC: 0.7284, Brier: 0.1672) and Gradient Boosting (0.8205, PR-AUC: 0.7159, Brier: 0.1701).
- **Artifact Selection:** HistGradientBoostingClassifier was saved as the non-linear implementation candidate for the screening artifact (`models/nhanes_screening_core.joblib`). It is not claimed to be objectively superior to Logistic Regression; rather, Logistic Regression serves as the linear benchmark comparator.

---

## 3. NHANES Experiment B: Full 58-Feature Configuration
Predictor Battery: All 58 candidate non-invasive features documented in the audited registry.

| Model | CV_ROC_AUC_Mean | CV_ROC_AUC_Std | CV_PR_AUC_Mean | CV_PR_AUC_Std | CV_Brier_Mean | Test_ROC_AUC | Test_PR_AUC | Test_Brier | Test_Acc_Def (0.50) | Test_Sens_Def (0.50) | Test_Spec_Def (0.50) | Selected_Threshold (Tau) | Test_Acc_Opt | Test_Sens_Opt | Test_Spec_Opt | Test_F1_Opt | Survey_Weighted_ROC_AUC | Survey_Weighted_Sens_Opt | Survey_Weighted_Spec_Opt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic_Regression | 0.8132 | 0.008 | 0.7076 | 0.0188 | 0.1773 | 0.8377 | 0.7396 | 0.1635 | 0.7628 | 0.7494 | 0.7709 | 0.45000000000000007 | 0.7558 | 0.7973 | 0.7304 | 0.7128 | 0.8415 | 0.6959 | 0.8065 |
| Random_Forest | 0.8168 | 0.0106 | 0.707 | 0.0176 | 0.1752 | 0.8282 | 0.7178 | 0.1681 | 0.755 | 0.754 | 0.7556 | 0.4700000000000001 | 0.7619 | 0.8041 | 0.736 | 0.7197 | 0.8322 | 0.7217 | 0.8085 |
| Gradient_Boosting | 0.8137 | 0.0073 | 0.7054 | 0.0152 | 0.1753 | 0.8224 | 0.7157 | 0.1682 | 0.7429 | 0.7608 | 0.7318 | 0.4600000000000001 | 0.7368 | 0.7882 | 0.7053 | 0.6948 | 0.8297 | 0.7092 | 0.7921 |

### Key Findings:
- **Observed Core vs. Full Difference:** Expanding from 24 core features to all 58 candidate predictors resulted in an observed change in test ROC-AUC of +0.0028 for Logistic Regression (0.8349 to 0.8377), -0.0007 for Random Forest (0.8289 to 0.8282), and +0.0019 for Gradient Boosting (0.8205 to 0.8224).
- **Scientific Caution:** Because these numerical differences are small and within sampling variability, we make no claims of formal statistical equivalence or clinical non-inferiority without prospective hypothesis testing. The parsimonious 24-feature set is preferred for clinical feasibility and reduced participant burden.

---

## 4. Framingham Experiment: 10-Year Incident CVD Risk
Evaluated on the primary prevention cohort (N = 3,983, 9.26% 10-year CVD events, 74 test positives).

| Model | CV_ROC_AUC_Mean | CV_ROC_AUC_Std | CV_PR_AUC_Mean | CV_PR_AUC_Std | CV_Brier_Mean | Test_ROC_AUC | Test_PR_AUC | Test_Brier | Test_Acc_Def (0.50) | Test_Sens_Def (0.50) | Test_Spec_Def (0.50) | Selected_Threshold (Tau) | Test_Acc_Opt | Test_Sens_Opt | Test_Spec_Opt | Test_F1_Opt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic_Regression | 0.7823 | 0.0282 | 0.3128 | 0.0489 | 0.1889 | 0.7648 | 0.2424 | 0.1955 | 0.6888 | 0.7027 | 0.6874 | 0.4000000000000001 | 0.591 | 0.8108 | 0.5685 | 0.2691 |
| Random_Forest | 0.7614 | 0.0218 | 0.2939 | 0.0668 | 0.1279 | 0.7528 | 0.2304 | 0.1377 | 0.8068 | 0.5135 | 0.8368 | 0.27 | 0.5659 | 0.7838 | 0.5436 | 0.2511 |
| Gradient_Boosting | 0.7475 | 0.0244 | 0.3035 | 0.0591 | 0.1276 | 0.7246 | 0.2071 | 0.1385 | 0.798 | 0.4189 | 0.8368 | 0.18000000000000002 | 0.5144 | 0.8649 | 0.4786 | 0.2485 |

### Key Findings:
- **Model Performance:** L2-regularized Logistic Regression (evaluated on standard_bp to prevent collinearity with derived metrics) achieved superior 5-fold cross-validation ROC-AUC (0.7823 +/- 0.0282) and held-out test discrimination (ROC-AUC: 0.7648, PR-AUC: 0.2424) compared to tree ensembles (Random Forest test ROC-AUC: 0.7528; Gradient Boosting test ROC-AUC: 0.7246).
- **Threshold Tuning & Recovery:** Tree models suffered severe sensitivity collapse at default tau = 0.50 (Sens: 41.89% - 51.35%). Out-of-fold threshold optimization successfully tuned decision thresholds (tau = 0.40 for Logistic Regression, tau = 0.27 for Random Forest, tau = 0.18 for Gradient Boosting) to achieve 81.08%, 78.38%, and 86.49% test sensitivity, respectively.
- **Methodological Limitations:** Formulated as a conditional landmark binary classifier on participants with complete 10-year follow-up or prior event. This is NOT a competing-risk survival model and is NOT a clinically validated replacement for established calculators such as the ACC/AHA Pooled Cohort Equations.

---

## 5. Survey-Weighted NHANES Evaluation Note
- Survey-weighted test metrics were computed using NHANES Mobile Examination Center 2-year weights (WTMEC2YR / `survey_weight_mec_2yr`).
- These metrics represent weighted test-sample estimates adjusting for unequal selection probabilities and examination non-response.
- Survey weighting adjusts for survey design within the sample, but does NOT establish external population validity or prospective clinical generalizability.

---

## 6. Serialized Model Artifacts
- `models/nhanes_screening_core.joblib` (Selected implementation candidate pipeline: HistGradientBoosting)
- `models/nhanes_screening_full.joblib` (Selected full candidate pipeline: HistGradientBoosting)
- `models/framingham_risk.joblib` (Selected primary risk pipeline: LogisticRegression)
- `models/model_metadata.json` (Consolidated parameter, threshold, OOF, and test performance registry)