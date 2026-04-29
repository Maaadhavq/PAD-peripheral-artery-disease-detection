# PAD Prediction Model — Welldoc Interview Guide

> Use this guide to answer any technical or behavioural question about this project in your Welldoc interview.
> Welldoc builds AI-driven chronic disease management tools (BlueStar for diabetes). PAD is a vascular disease that co-occurs in **30–40% of diabetics** — this project is directly relevant to their mission.

---

## 1. The 30-Second Pitch

> *"I built an end-to-end ML pipeline to detect Peripheral Artery Disease from real ICU clinical records — MIMIC-IV. I pulled labs, medications, diagnoses, and demographics, engineered 13 clinically meaningful features, fixed five data leakage bugs I found in my own code, and benchmarked six classifiers. The best model hit AUC 0.88. The whole thing is reproducible in a single Colab notebook."*

---

## 2. What Is PAD and Why Does It Matter to Welldoc?

- **PAD** = narrowing of peripheral arteries (usually legs) due to atherosclerosis.
- Risk factors: **diabetes, hypertension, smoking, high cholesterol** — the same population Welldoc's BlueStar platform serves.
- **30–40% of type-2 diabetics develop PAD**, often undetected until limb-threatening ischemia.
- Early ML-based flagging from routine EHR data = the exact kind of preventive clinical AI Welldoc builds.

---

## 3. Data: MIMIC-IV

| Table used | Why |
|---|---|
| `admissions.csv` | Admission timestamps, subject IDs |
| `patients.csv` | Age, gender, anchor year |
| `diagnoses_icd.csv` | PAD ICD codes + comorbidities |
| `d_labitems.csv` | Map lab item IDs → names |
| `labevents.csv` | Actual lab values (chunked, 1M rows at a time) |
| `prescriptions.csv` | Medication usage (chunked) |

**PAD ICD codes used:**
- ICD-10: `I739, I7021, I7022, I7029, I708, I709, I7391, I7399`
- ICD-9: `4439, 44389, 44381, 44022, 44029, 4408, 4409`

---

## 4. Feature Engineering (13 Features)

| Category | Features | Clinical reasoning |
|---|---|---|
| **Demographics** | age at admission, gender (binary) | Older males have higher PAD prevalence |
| **Lab results** | cholesterol, glucose, creatinine, hemoglobin, platelet count | Cholesterol → atherosclerosis; glucose → diabetes proxy; creatinine → renal/vascular; hemoglobin → anemia; platelets → clotting |
| **Comorbidities** | diabetes, hypertension, heart disease, stroke history | All share the same vascular risk pathway as PAD |
| **Medications** | statin use, antiplatelet use (aspirin/clopidogrel) | Statins prescribed for atherosclerotic risk; antiplatelets for clot prevention |

---

## 5. The 5 Data Leakage Bugs I Fixed

This is the most impressive part to discuss in interviews — it shows rigour.

### Fix 1 — Matched Controls (Confounding)
**Problem:** Controls were randomly sampled regardless of age/gender. The PAD group was mostly older males; controls were random → model learned "old male = PAD" rather than clinical features.  
**Fix:** Matched controls 1:1 on age bracket × gender using `groupby` + `random.sample`.

### Fix 2 — Temporal Leakage in Demographics
**Problem:** `admittime` was used for feature engineering and then dropped, but the column rename was missing — future admissions could bleed into the feature set.  
**Fix:** Renamed to `admittime_index` explicitly, kept it through comorbidity/medication filtering, then dropped before modelling.

### Fix 3 — Duplicate Lab Columns
**Problem:** Multiple `itemid`s can map to the same lab label (e.g., point-of-care glucose vs. lab glucose). Pivoting before label-mapping created duplicate columns that got silently dropped.  
**Fix:** Mapped `itemid → label` *before* groupby aggregation → duplicates averaged rather than dropped.

### Fix 4 — Temporal Leakage in Comorbidities
**Problem:** A patient diagnosed with diabetes *after* their PAD admission was still flagged as `has_diabetes = 1` — the future diagnosis leaked into the feature.  
**Fix:** Joined diagnoses with their own `admittime`; only flagged a comorbidity if `diag_admittime ≤ admittime_index`.

### Fix 5 — Temporal Leakage in Medications
**Problem:** Same issue with prescriptions — future prescriptions (post-PAD-admission) were counted.  
**Fix:** Joined prescriptions with `presc_admittime`; only counted medications prescribed at or before the index admission.

---

## 6. Patient-Level Train/Test Split

**Standard `train_test_split` is wrong here** — if a patient has multiple admissions, one can end up in train and another in test, causing data leakage.

**Fix:** Used `GroupShuffleSplit(groups=subject_id)` — guaranteed no patient appears in both sets.

```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))
```

---

## 7. Class Imbalance Handling

PAD cases are a minority in any hospital cohort. Three different strategies used across models:

| Model | Strategy |
|---|---|
| Logistic Regression, RF, SVM | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight = (neg_count / pos_count)` |
| LightGBM | `class_weight='balanced'` |
| MLP | Manual oversampling via `sklearn.utils.resample` (MLP has no class_weight param) |

---

## 8. Models Benchmarked

| Model | Notes |
|---|---|
| Logistic Regression | Baseline; interpretable coefficients |
| Random Forest | Handles non-linearity, good with mixed features |
| SVM | Strong on small-medium tabular data |
| MLP | Non-linear but needs more data to shine |
| XGBoost | Often best on structured clinical data |
| LightGBM | Faster XGBoost variant; good AUC |

**Primary metric: AUC-ROC** — chosen because it measures discrimination across all thresholds, which matters in a clinical setting where the threshold is tunable based on cost of false negatives.

**Best model achieved AUC 0.88.**

---

## 9. Why AUC Over Accuracy?

Class imbalance makes accuracy misleading — a model predicting "no PAD" for everyone could get 90%+ accuracy.  
AUC measures *rank discrimination*: can the model score true PAD patients higher than controls? It's threshold-independent, which matters because in clinical deployment you'd tune the threshold based on the cost of missing a PAD case (high recall) vs. unnecessary referrals (high precision).

---

## 10. Processing Large Files Without Crashing

`labevents.csv` and `prescriptions.csv` from MIMIC-IV are multi-GB files. Strategy used:

```python
for chunk in pd.read_csv('labevents.csv.gz', chunksize=1_000_000):
    filtered = chunk[chunk['subject_id'].isin(our_patients)]
    chunks.append(filtered)
```

Filter while reading → only keep rows for patients in our cohort → concatenate filtered chunks. This avoids loading the full file into RAM.

---

## 11. Preprocessing Pipeline

```
raw MIMIC-IV tables
    → cohort identification (ICD codes)
    → 1:1 age/gender matched controls
    → feature engineering (demographics + labs + comorbidities + meds)
    → temporal leakage prevention (all features restricted to pre-index time)
    → SimpleImputer (mean) for missing labs
    → StandardScaler
    → GroupShuffleSplit (patient-level)
    → 6 classifiers trained + compared by AUC
```

---

## 12. Questions You'll Definitely Be Asked

**Q: What is PAD and why did you choose it?**  
> PAD is peripheral artery disease — narrowed arteries limiting blood flow to limbs, usually legs. I chose it because it's underdiagnosed, strongly linked to diabetes and hypertension, and MIMIC-IV had enough coded cases to build a meaningful cohort. The data availability + clinical relevance made it a good fit.

**Q: What was your biggest technical challenge?**  
> Temporal leakage. It took me several iterations to realise that simply joining diagnoses and medications without time-restricting them meant future clinical events were bleeding into the feature set. I had to redesign the comorbidity and medication pipeline to filter by admission timestamp before the index event.

**Q: Why not use deep learning here?**  
> With tabular EHR data at this scale (thousands of patients, 13 features), tree-based models and logistic regression are strong baselines and often outperform MLPs. I did include an MLP in the benchmark. For this dataset, boosting models won. Deep learning would make more sense with raw notes (then you'd use BERT-style models) or imaging data.

**Q: How would you deploy this?**  
> The sklearn pipeline (imputer + scaler + trained model) can be serialized with `joblib` and wrapped in a Flask or FastAPI endpoint. Input = a JSON object with the 13 feature values; output = PAD probability score. In a production healthcare setting you'd add input validation, audit logging, and a calibration step so probabilities are clinically meaningful.

**Q: How is this relevant to Welldoc?**  
> Welldoc manages chronic conditions including diabetes. PAD affects 30–40% of diabetics and is often missed until it's serious. The same feature engineering pipeline — pulling from EHR tables, flagging comorbidities, processing lab values, applying ML risk scoring — is directly the kind of work that powers Welldoc's risk stratification engine. I've already handled the hard parts: leakage prevention, class imbalance, patient-level splitting.

**Q: What would you improve next?**  
> Three things: (1) cross-validation instead of a single split for more reliable AUC estimates; (2) SHAP values for feature importance to understand which clinical signals are driving predictions; (3) calibration (Platt scaling or isotonic regression) so the model's probability outputs are actually well-calibrated for clinical use.

**Q: How did you handle missing lab values?**  
> Mean imputation via `SimpleImputer`. It's a reasonable baseline for clinical labs where missingness is often MCAR or MAR (e.g., the test wasn't ordered, not because the patient had an unusual value). A better approach for a production system would be MICE (multiple imputation) or a missingness indicator feature.

**Q: What's GroupShuffleSplit and why use it instead of train_test_split?**  
> GroupShuffleSplit ensures no patient appears in both train and test sets. If a patient had 3 admissions and one ended up in train and another in test, the model could effectively "memorise" that patient's physiology. GroupShuffleSplit splits on `subject_id` groups, preventing this.

**Q: Explain the cohort identification process.**  
> I pulled all admissions where the patient had a PAD ICD-9 or ICD-10 code in `diagnoses_icd`. That became the case group. For controls, I sampled admissions from patients with no PAD code, matched 1:1 on age bracket and gender to prevent demographic confounding. This gives a balanced, demographically-matched cohort.

---

## 13. Welldoc-Specific Angles

| Welldoc cares about | How this project demonstrates it |
|---|---|
| Chronic disease AI | PAD is a chronic vascular condition; same risk factors as Welldoc's patient population |
| EHR data processing | Built an end-to-end MIMIC-IV pipeline from raw tables |
| Clinical feature engineering | 13 clinically reasoned features, not just everything thrown in |
| Data quality & leakage | Found and fixed 5 separate leakage issues — shows production-level rigour |
| Risk stratification | AUC-based model outputs a risk score that's threshold-tunable |
| Reproducibility | Single notebook, seeded splits, saved CSV for downstream reuse |

---

## 14. One-Line Answers for Rapid-Fire Questions

| Question | Answer |
|---|---|
| Dataset | MIMIC-IV (PhysioNet, credentialed access) |
| Cohort size | PAD cases matched 1:1 with controls |
| Best AUC | 0.88 |
| Best model | LightGBM / XGBoost (gradient boosted trees) |
| Number of features | 13 |
| Missing value strategy | Mean imputation (SimpleImputer) |
| Imbalance strategy | class_weight='balanced', scale_pos_weight, oversampling for MLP |
| Split strategy | GroupShuffleSplit by subject_id (patient-level) |
| Language & libs | Python, pandas, scikit-learn, XGBoost, LightGBM, matplotlib |
| Runtime | Google Colab, Google Drive for large files |
