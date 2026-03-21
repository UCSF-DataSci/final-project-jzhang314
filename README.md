# Predicting 30-Day Hospital Readmission Using Synthetic EHR Data

## Problem Overview

Hospital readmissions within 30 days of discharge are a key quality metric in US healthcare. The CMS Hospital Readmissions Reduction Program penalizes hospitals with excessive rates, with cumulative penalties exceeding $1.9 billion since 2012. This project builds a machine learning pipeline to predict which patients are at risk of 30-day readmission using structured EHR data, enabling targeted interventions such as follow-up calls and discharge planning.

## Dataset

**Source:** [Synthea](https://synthetichealth.github.io/synthea/) synthetic patient generator (1K sample, CSV format) [1]

**Dimensions:** 1,171 patients, 53,346 encounters, 299,697 observations, 42,989 medications, 34,981 procedures

**Cohort:** 3,928 inpatient/emergency discharges (786 readmissions, 20% rate)

**Outcome:** Binary — readmitted within 30 days (1) or not (0). Requires at least 1 day gap to exclude same-day transfers.

**Input Features (26 total):**

| Category | Features |
|---|---|
| Demographics | Age, gender, encounter type (inpatient vs emergency) |
| Length of stay | Days between admission and discharge |
| Prior conditions | Number of unique diagnoses before admission |
| Comorbidity flags | CHF, COPD, diabetes, CKD, hypertension, cancer, depression, asthma |
| Prior medications | Number of unique medications before admission |
| Prior procedures | Number of unique procedures before admission |
| Lab values | Most recent pre-admission BMI, systolic/diastolic BP, heart rate, body weight, glucose, HbA1c |
| Utilization history | Visit counts in prior 6 months and 1 year, ED and inpatient subtotals |

All clinical features use only data recorded before the index admission to prevent data leakage.

## How to Run

**Requirements:**
```
pip install polars pyarrow scikit-learn xgboost tensorflow matplotlib pandas
```

**Steps:**
1. Download Synthea 1K CSV sample from https://synthetichealth.github.io/synthea/
2. Extract the CSV files to a folder
3. Update `DATA_DIR` in the notebook to point to your CSV folder
4. Run `readmission_project_final.ipynb` top to bottom (restart kernel first)

**Note:** The prior visit history cell (Section 11h) takes several minutes due to row-by-row iteration. All other cells run in seconds.

## Decisions and Trade-offs

- **Synthetic data over real EHR:** Used Synthea because MIMIC-IV access requires PhysioNet credentialing (days-long process). Trade-off: model performance is inflated compared to real-world data (literature reports AUC 0.60-0.75 on real EHR).
- **Data leakage prevention:** Initial version used encounter-level features (during hospitalization). Rewrote all joins to use only pre-admission data. This lowered AUC but produced methodologically sound results.
- **Removed redundant features:** Dropped total counts (n_conditions, n_medications, n_procedures) due to multicollinearity with unique counts. Trade-off: lost some granularity but gained interpretable logistic regression coefficients.
- **Patient-level train/test split:** Used GroupShuffleSplit by patient ID instead of row-level splitting. Prevents the model from memorizing patient-specific patterns. Trade-off: uneven class distribution between train (18.8%) and test (24.4%).
- **Class weighting over threshold tuning:** Used class_weight="balanced" during training rather than adjusting decision thresholds. Trade-off: tree-based models (RF, XGBoost) still show low recall at 0.5 threshold due to a known probability calibration issue with ensemble methods.
- **No cross-validation for neural network:** Keras does not integrate with sklearn's cross_val_score. Trade-off: NN comparison is based on a single train/validation split, making it not strictly equivalent to the 5-fold CV used for classical models.
- **Median imputation for missing labs:** Simple but loses the informativeness of missingness. In real clinical data, a missing lab often signals clinical judgment (test not ordered because not indicated).

## Example Output

**Cross-validation AUC:**
```
Logistic Regression: 0.873 ± 0.019
Random Forest:       0.858 ± 0.026
XGBoost:             0.876 ± 0.027
```

**Test set results (threshold = 0.5):**

| Model | Test AUC | Recall | Precision |
|---|---|---|---|
| Logistic Regression | 0.794 | 55% | 51% |
| Random Forest | 0.839 | 7% | 52% |
| XGBoost | 0.798 | 10% | 54% |
| Neural Network | 0.735 | 57% | 56% |

**Top predictive features (Random Forest):** Prior visits in 6 months, prior visits in 1 year, age, prior inpatient admissions, number of unique conditions, cancer history.

The pipeline also produces ROC curves, confusion matrices, feature importance plots, and logistic regression coefficient charts.

## Citations

[1] Walonoski J, et al. Synthea: An approach, method, and software mechanism for generating synthetic patients and the synthetic electronic health care record. JAMIA. 2018;25(3):230-238. https://doi.org/10.1093/jamia/ocx079

[2] Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data. PMC. 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/

[3] Building Prediction Models for 30-Day Readmissions Among ICU Patients Using Both Structured and Unstructured Data in Electronic Health Records. PMC. 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11271049/
