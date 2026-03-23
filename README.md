# PAD Prediction Model v2.0

> 🚧 **Note:** This project is currently undergoing development and is a work in progress.

A machine learning pipeline for predicting **Peripheral Artery Disease (PAD)** using clinical data from [MIMIC-IV](https://physionet.org/content/mimiciv/). Built this as a way to explore how well standard ML classifiers can pick up on PAD from routine hospital data — labs, demographics, comorbidities, and medications.

## What it does

Takes raw MIMIC-IV tables and builds a binary classifier that predicts whether a patient has PAD. The pipeline handles everything end-to-end:

- Pulls admissions, diagnoses, lab events, and prescriptions from MIMIC-IV
- Identifies PAD patients via ICD-9/ICD-10 codes and creates a matched control group
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

## Data

Uses [MIMIC-IV](https://physionet.org/content/mimiciv/) which requires PhysioNet credentialed access. You'll need these files in your Google Drive:

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

1. Upload the MIMIC-IV files to `My Drive/mimic_data/` on Google Drive
2. Open `pad_model.ipynb` in [Google Colab](https://colab.research.google.com/)
3. Run all cells — it'll mount your Drive, process the data, and train the models
4. The processed dataset gets saved as `pad_model_dataset.csv` so you can skip the heavy preprocessing next time

## Dependencies

All available in Colab by default except the boosting libraries:

```
pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, lightgbm
```

Install the extras with:
```bash
pip install xgboost lightgbm
```

## Project structure

```
pad2.0/
├── pad_model.ipynb      # main notebook — data processing + model training
└── README.md
```

## Notes

- The lab events and prescriptions files are large, so the notebook processes them in chunks to avoid memory issues
- Class imbalance is handled through balanced class weights and `scale_pos_weight` for boosting models
- The control group is randomly sampled to match the PAD group size, so results may vary slightly between runs (random seed is set for the train/test split though)

## License

This project uses MIMIC-IV data which is subject to the [PhysioNet Credentialed Health Data Use Agreement](https://physionet.org/content/mimiciv/). Make sure you have proper access before using the data.
