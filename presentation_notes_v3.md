# Predicting 30-Day Hospital Readmission Using Synthetic EHR Data
## Presentation Notes

---

## Slide 1: Problem Statement

Hospital readmissions within 30 days of discharge are one of the biggest headaches in healthcare right now. Not only do they signal that something went wrong with the patient's care, but they also cost a ton of money. In the US, the CMS Hospital Readmissions Reduction Program penalizes hospitals with excessive readmission rates — since the program started in 2012, hospitals have experienced nearly $1.9 billion in cumulative penalties, with up to 3% of total Medicare payments withheld from the worst performers.

The idea behind this project is straightforward: if we can predict which patients are likely to come back within 30 days, hospitals can intervene early — follow-up calls, better discharge planning, medication check-ins — and hopefully prevent that return visit. In other words, we are trying to build a machine learning pipeline that flags high-risk patients before they walk out the door.

---

## Slide 2: Dataset

For this project I used Synthea, which is an open-source synthetic patient generator developed by MITRE Corporation [1]. It generates realistic but not real patient data, so there are no privacy concerns to worry about. The dataset I worked with has 1,171 synthetic patients with full medical histories — over 53,000 encounters, 300,000 lab observations, 43,000 medication records, and 35,000 procedures. All stored in relational CSV tables that mirror how real EHR systems are structured.

From those 53,000 encounters, I filtered down to inpatient and emergency visits specifically, which gave me 3,928 eligible discharges. Out of those, 786 resulted in a readmission within 30 days — that is a 20% readmission rate. This is actually somewhat close to real-world numbers (the national average hovers around 15-20%), which is a good sign that Synthea is generating reasonable data.

One thing to keep in mind though: Synthea generates patients through programmed state-transition modules based on published CDC statistics. So the relationships between variables are essentially coded in, not discovered from nature. This means model performance would likely be lower on real data — published studies using real EHR typically report AUCs around 0.60-0.75.

---

## Slide 3: Existing Work

The approach I took here is well-established in the readmission prediction literature. Two papers that closely align with my methods:

First, a 2025 study from PMC that predicted 30-day readmissions in diabetes patients using the same structured EHR features I used — demographics, comorbidities, medication burden, prior hospitalizations [2]. They compared logistic regression, random forest, XGBoost, and a deep neural network, and found XGBoost performed best with an AUC of 0.667.

Second, a study using MIMIC-III (real ICU data) that built prediction models using demographics, lab tests, and comorbidities [3]. Their best model achieved AUC of 0.757 with logistic regression.

Both of these studies define readmission the same way I did and use similar feature engineering approaches. My project essentially replicates this established pipeline but on synthetic data, with the goal of demonstrating the methodology rather than producing a clinical tool.

---

## Slide 4: Tools and Methods

This project pulls from pretty much everything we learned in class: SQL for cohort building (loaded CSVs into SQLite, used window functions to identify readmissions), Polars for feature engineering (joins, group-bys, date arithmetic, pivoting lab values), scikit-learn for classical ML (logistic regression, random forest, cross-validation, ROC/AUC), XGBoost as a gradient boosting comparison, TensorFlow/Keras for the dense neural network, and matplotlib for all the visualizations.

---

## Slide 5: Feature Engineering

This is where most of the thinking went. I engineered 26 features across six categories, and the critical design decision was making sure every clinical feature only uses data from before the patient's admission. This prevents data leakage, which is when information that would not be available at prediction time accidentally gets into the model.

The features include: age, gender, whether the visit was emergency or inpatient, length of stay, number of unique prior diagnoses/medications/procedures, binary flags for 8 high-risk comorbidities (CHF, COPD, diabetes, CKD, hypertension, cancer, depression, asthma), most recent lab values before admission (BMI, blood pressure, heart rate, glucose, body weight, HbA1c), and utilization history (visit counts in prior 6 months and 1 year).

For missing lab values — not every patient has every lab test recorded — I imputed with training-set medians. Importantly, this imputation was done after the train/test split to avoid leaking test set information into the imputation.

---

## Slide 6: Issues Overcome Along the Way

This project went through several rounds of critical review and revision, and honestly the debugging process taught me more than the initial build.

The biggest issue was data leakage. My first version joined clinical features by encounter ID, which meant I was using diagnoses and medications from the current hospitalization to predict whether that same hospitalization would lead to a readmission. In other words, I was letting the model peek at information it should not have had. The fix was rewriting all feature joins to use patient ID with a date filter — only records before the admission date.

Second issue: the comorbidity flags for CHF and asthma were returning 0% prevalence despite these conditions existing in the data. It turns out Synthea uses "Chronic congestive heart failure" (lowercase c) and "Childhood asthma" (lowercase a), while my keyword matching was case-sensitive. A small bug that completely wiped out two features. I fixed it with case-insensitive regex.

Third: I initially had both total counts (n_conditions) and unique counts (n_unique_conditions) as features. These are highly correlated — more conditions generally means more unique conditions. This multicollinearity made the logistic regression coefficients uninterpretable: one came out positive and the other negative for essentially the same concept. I dropped the total counts and kept only unique counts.

Fourth: my original train/test split was at the row level, not the patient level. This means the same patient could have encounters in both train and test, letting the model memorize patient-specific patterns. I switched to a grouped split by patient ID.

---

## Slide 7: Class Imbalance

With 80% non-readmission and 20% readmission in our cohort, a model that just predicts "no readmission" for everyone gets 80% accuracy while being completely useless clinically. This is a well-known problem in healthcare ML.

To address this, all models were trained with class_weight="balanced" (or equivalent for XGBoost). What this does is essentially tell the model to pay more attention when it gets a readmission case wrong. Logistic regression adjusts its coefficients, random forest adjusts its bootstrap sampling, and XGBoost adjusts its gradient calculations.

One interesting thing I found: even with class weighting, tree-based models (RF and XGBoost) still showed very low recall at the default 0.5 decision threshold. This is a known calibration issue — ensemble methods like random forest average predictions across 100 trees, which compresses the predicted probabilities toward the middle. So even if a patient is high-risk, the probability might come out as 0.35 instead of crossing 0.5. The model ranks patients correctly (high AUC) but the yes/no cutoff does not reflect that ranking well. This is called the calibration vs. discrimination tradeoff, and it is a real challenge in clinical ML.

---

## Slide 8: Results

Here is the summary across all four models:

Logistic Regression: CV AUC 0.873, Test AUC 0.794, Readmit Recall 55%
Random Forest: CV AUC 0.858, Test AUC 0.839, Readmit Recall 7%
XGBoost: CV AUC 0.876, Test AUC 0.798, Readmit Recall 10%
Neural Network: Test AUC 0.735, Readmit Recall 57%

The top predictive features from random forest were: number of visits in prior 6 months, visits in prior year, age, prior inpatient admissions, number of unique conditions, and cancer history. These align well with what is established in the readmission literature — utilization history and comorbidity burden are consistently the strongest predictors.

From logistic regression coefficients, the biggest positive predictors were cancer history, number of unique conditions, male gender, recent visit frequency, and CKD. These make clinical sense — sicker patients with more complex histories and frequent healthcare contact are at higher risk.

---

## Slide 9: Discussion

The most interesting finding here is actually the split between models that rank well (RF and XGBoost with high AUC) versus models that make better yes/no decisions (LR and NN with better recall). AUC tells you the model can distinguish between high-risk and low-risk patients, but recall tells you how many actual readmissions it catches. These are fundamentally different questions with different clinical implications.

The neural network achieved the most balanced performance overall (recall 57%, precision 56%, F1 0.57) but had the lowest AUC. This is a well-documented finding in the ML literature — on small tabular datasets with around 3,000 rows, tree-based methods almost always outperform deep learning. Neural networks need much larger datasets or high-dimensional inputs like images and text to show their advantage. Adding more layers would just increase overfitting.

With those being said, Random Forest had the highest test AUC (0.839) and the smallest gap between CV AUC and test AUC, meaning it generalizes best to unseen patients. However, its low recall at the 0.5 threshold means probability calibration would be needed before clinical deployment.

---

## Slide 10: Limitations and Future Work

There are several limitations worth acknowledging. First, this is synthetic data — the variable relationships are programmed, not natural. Real-world performance would drop. Second, the same patient can contribute multiple encounters within the training set, which violates independence assumptions. Third, median imputation for missing labs is simplistic — in real clinical data, a missing lab often means the clinician did not think it was needed, which is informative in itself. Fourth, the neural network was not cross-validated like the classical models, making that comparison not perfectly fair.

For future work, the most natural next step would be applying this exact pipeline to MIMIC-IV, which is real ICU data. The code is built to be transferable — you would just swap the data source and adjust column names. Other improvements could include probability calibration for the tree models, SHAP analysis for better interpretability, incorporating clinical notes via NLP embeddings, and exploring threshold optimization rather than using the default 0.5.

Overall, this project demonstrates a complete, methodologically sound ML pipeline for readmission prediction — from SQL cohort building through feature engineering to model comparison. The pipeline itself is the contribution, not the specific AUC numbers.

---

## References

[1] Walonoski J, et al. Synthea: An approach, method, and software mechanism for generating synthetic patients and the synthetic electronic health care record. JAMIA. 2018;25(3):230-238. https://doi.org/10.1093/jamia/ocx079

[2] Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data. PMC. 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/

[3] Building Prediction Models for 30-Day Readmissions Among ICU Patients Using Both Structured and Unstructured Data in Electronic Health Records. PMC. 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11271049/
