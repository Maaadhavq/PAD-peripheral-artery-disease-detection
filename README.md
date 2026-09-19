# PAD Prediction Model v2.0

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maaadhavq/PAD-peripheral-artery-disease-detection/blob/main/pad_model.ipynb)

> 🚧 **Note:** This project is currently undergoing development and is a work in progress.

A machine learning pipeline for predicting **Peripheral Artery Disease (PAD)** using clinical data from [MIMIC-IV](https://physionet.org/content/mimiciv/). Built this as a way to explore how well standard ML classifiers can pick up on PAD from routine hospital data — labs, demographics, comorbidities, and medications.

## What it does

Takes raw MIMIC-IV tables and builds a binary classifier that predicts whether a patient has PAD. The pipeline handles everything end-to-end:

- Pulls admissions, diagnoses, lab events, and prescriptions from MIMIC-IV
- Identifies PAD patients via ICD-9/ICD-10 codes and samples a 1:1 control group matched on age bracket and gender
- Engineers features from five categories:
  - **Demographics** — age at admission, gender
  - **Lab results** — cholesterol, glucose, creatinine, hemoglobin, platelet count
  - **Comorbidities** — diabetes, hypertension, heart disease, stroke history
  - **Medications** — statin and antiplatelet usage
- Trains and compares six models:
  - Logistic Regression
  - Random Forest
  - SVM
  - MLP (Neural Network)
  - XGBoost
  - LightGBM
- Picks the best one by AUC and shows confusion matrix + ROC curve

## Avoiding leakage

Most of the work in this notebook went into making sure the model can't cheat:

- **Patient-level cohort** — controls are drawn from patients with *no* PAD code in any admission, so the same person never appears as both a case and a control
- **Patient-level split** — `GroupShuffleSplit` on `subject_id`, so a patient with multiple admissions can't land in both train and test
- **No future information** — comorbidities and medications only count if they were recorded at or before the index admission; labs come from the index admission only
- **Matched controls** — 1:1 on age bracket × gender, so the model has to learn from clinical features rather than "older male = PAD"
- **Lab de-duplication** — multiple `itemid`s that map to the same lab label (e.g. point-of-care vs. lab glucose) are averaged rather than silently dropped

## Data

Uses [MIMIC-IV](https://physionet.org/content/mimiciv/) which requires PhysioNet credentialed access. You'll need these files:

```
mimic_data/
├── admissions.csv.gz
├── patients.csv.gz
├── diagnoses_icd.csv.gz
├── d_labitems.csv.gz
├── labevents.csv.gz
└── prescriptions.csv.gz
```

## How to run

**On Colab (recommended):**

1. Upload the MIMIC-IV files to `My Drive/mimic_data/` on Google Drive
2. Open `pad_model.ipynb` in [Google Colab](https://colab.research.google.com/)
3. Run all cells — it'll mount your Drive, process the data, and train the models

**Locally:**

1. Put the MIMIC-IV files in a `mimic_data/` folder next to the notebook
2. `pip install -r requirements.txt`
3. Run the notebook — it detects it's not on Colab and reads from `./mimic_data/` instead

Either way, the processed dataset gets saved as `pad_model_dataset.csv` so you can skip the heavy preprocessing next time.

## Dependencies

Everything is preinstalled on Colab. For a local run:

```bash
pip install -r requirements.txt
```

## Project structure

```
├── pad_model.ipynb      # main notebook — data processing + model training
├── requirements.txt
└── README.md
```

## Notes

- The lab events and prescriptions files are large, so the notebook processes them in chunks to avoid memory issues
- Class imbalance is handled through balanced class weights, `scale_pos_weight` for XGBoost, and minority oversampling for the MLP (which has no class-weight option)
- Control sampling and the train/test split are both seeded, so runs are reproducible
- Missing lab values are mean-imputed (fit on the training set only)

## What's next

- Cross-validation instead of a single split for a more reliable AUC estimate
- SHAP values to see which clinical signals drive predictions
- Probability calibration (Platt / isotonic) before any clinical use

## License

This project uses MIMIC-IV data which is subject to the [PhysioNet Credentialed Health Data Use Agreement](https://physionet.org/content/mimiciv/). Make sure you have proper access before using the data.
