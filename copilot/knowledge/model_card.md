# Model card — PAD risk model

## What this model is

A binary classifier that estimates the probability that a hospital admission
belongs to a patient with peripheral artery disease, using 13 routinely
collected features: demographics, five lab values, four comorbidity flags and
two medication flags.

Six classifiers are compared — logistic regression, random forest, SVM, a small
MLP, XGBoost and LightGBM. The one with the best cross-validated AUC is kept.
Live performance numbers for the current build are in `artifacts/model_card.json`,
which is written by the training notebook; this document describes the design
that produced them.

## What it is not

**This is not a diagnostic device and must not be used to make clinical
decisions.** It is a research project built on a retrospective ICU database.
It has never been validated prospectively, on any external cohort, or against
the actual diagnostic standard for PAD.

The clinical diagnosis of PAD rests on history, examination and the
ankle-brachial index — none of which this model uses. The model is trained
against ICD billing codes, which are a proxy for diagnosis, not diagnosis
itself.

## Data

MIMIC-IV, a deidentified critical care database from Beth Israel Deaconess
Medical Center, accessed under the PhysioNet Credentialed Health Data Use
Agreement. Tables used: `admissions`, `patients`, `diagnoses_icd`,
`d_labitems`, `labevents`, `prescriptions`.

## Cohort

Cases are admissions carrying a PAD ICD-9 (`4439, 44389, 44381, 44022, 44029,
4408, 4409`) or ICD-10 (`I739, I7021, I7022, I7029, I708, I709, I7391, I7399`)
code.

Controls are drawn from patients with **no PAD code in any admission**, matched
1:1 to cases on age bracket and gender, one admission per control patient.
Matching on admissions alone would let a PAD patient's other stays become
controls, putting the same person on both sides of the label; and without
demographic matching the model learns "older male" rather than anything
clinical.

## Guarding against leakage

- **Patient-level splitting.** Train and test are separated by `subject_id`
  using `GroupShuffleSplit`, so a patient with several admissions cannot appear
  on both sides and let the model memorise their physiology.
- **Model selection on cross-validation.** The winner is chosen by grouped
  5-fold CV on the training set. The held-out test set is scored once, after
  the choice is made, so the reported number is not the maximum over six
  attempts.
- **Preprocessing inside the pipeline.** Mean imputation and standardisation
  are pipeline steps, so they are refit within each CV fold and never see
  validation or test data.
- **Temporal restriction.** Comorbidity and medication features come only from
  admissions strictly before the index admission.

## Known limitations

- **ICD codes are the label, not a diagnosis.** Coding is driven by billing and
  documentation practice. Undercoded PAD patients sit in the control pool;
  miscoded ones inflate the case group.
- **ICU population.** MIMIC-IV patients are sicker than a general or primary
  care population, so these probabilities should not be read as population risk.
- **No ankle-brachial index, no smoking status, no imaging.** Smoking is among
  the strongest PAD risk factors and is not reliably available in the tables
  used here, so the model is blind to it.
- **Matched controls change the base rate.** The cohort is balanced by
  construction, so the output is a discrimination score rather than a
  calibrated population probability.
- **Probabilities are uncalibrated.** No Platt scaling or isotonic regression
  has been applied, so a reported 0.8 should not be read as an 80% chance.
- **Treatment flags may invert intuitively.** A patient on statins and
  antiplatelets already has recognised and treated cardiovascular risk, so
  these features can behave differently from naive expectation.

## Intended use

Education and demonstration: showing an end-to-end clinical ML pipeline with
explicit handling of leakage, class imbalance, and explanation. Any clinical
application would need prospective validation, calibration, prospective
external testing and regulatory review.
