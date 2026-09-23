"""Feature engineering: labs, comorbidities, medications.

Every feature here is restricted to information that exists at or before the
index admission. Two different cutoffs apply:

* **Labs** are taken from the index admission itself — they are the measurements
  available when the patient presents.
* **Comorbidities and medications** come from *strictly prior* admissions only.
  ICD codes are assigned at discharge, so a code on the index admission is not
  known at admission time; and statins/antiplatelets are the standard treatment
  for PAD, so counting prescriptions written during the index stay would feed
  the model a consequence of the diagnosis it is meant to predict.
"""

import pandas as pd

from pad.cohort import _matches_codes
from pad.config import COMORBIDITY_CODES, LAB_RENAME, LAB_TESTS, MEDICATION_PATTERNS
from pad.io import read_filtered_chunks


def _flag_prior_events(master_df, events, time_column, strict=True):
    """Flag index admissions that have a matching event before them.

    ``events`` holds (subject_id, <time_column>) pairs. Returns a 0/1 Series
    aligned to master_df.
    """
    index_times = master_df[["hadm_id", "subject_id", "admittime_index"]]
    merged = pd.merge(index_times, events, on="subject_id", how="left")
    if strict:
        hit = merged[time_column] < merged["admittime_index"]
    else:
        hit = merged[time_column] <= merged["admittime_index"]
    valid = set(merged[hit]["hadm_id"].dropna().unique())
    return master_df["hadm_id"].isin(valid).astype(int)


def add_labs(master_df, mimic_path, d_labitems, lab_tests=LAB_TESTS):
    """Mean lab values for the index admission, one column per lab label."""
    target = d_labitems[d_labitems["label"].isin(lab_tests)][["itemid", "label"]]

    labs = read_filtered_chunks(
        f"{mimic_path}/labevents.csv.gz",
        subject_ids=master_df["subject_id"].unique(),
        columns=["subject_id", "hadm_id", "itemid", "valuenum"],
        itemids=target["itemid"].tolist(),
    )
    if labs.empty:
        return master_df

    labs = labs.dropna(subset=["valuenum"])
    # Map itemid -> label before aggregating: several itemids can share a label
    # (point-of-care vs lab glucose). Mapping first averages them together
    # instead of producing duplicate columns that get silently dropped.
    labs = pd.merge(labs, target, on="itemid", how="inner")
    aggregated = labs.groupby(["hadm_id", "label"])["valuenum"].mean().reset_index()
    pivoted = aggregated.pivot(index="hadm_id", columns="label", values="valuenum").reset_index()
    pivoted.columns.name = None
    return pd.merge(master_df, pivoted, on="hadm_id", how="left")


def add_comorbidities(master_df, diagnoses, subject_admissions,
                      comorbidity_codes=COMORBIDITY_CODES):
    """Binary comorbidity flags from diagnoses on strictly prior admissions."""
    diagnoses_timed = pd.merge(
        diagnoses[["subject_id", "hadm_id", "icd_code", "icd_version"]],
        subject_admissions[["hadm_id", "admittime"]].rename(
            columns={"admittime": "diag_admittime"}
        ),
        on="hadm_id",
        how="inner",
    )

    out = master_df.copy()
    for disease, codes in comorbidity_codes.items():
        matched = diagnoses_timed[_matches_codes(diagnoses_timed, codes)]
        events = matched[["subject_id", "diag_admittime"]].drop_duplicates()
        out[disease] = _flag_prior_events(out, events, "diag_admittime", strict=True)
    return out


def add_medications(master_df, mimic_path, subject_admissions,
                    medication_patterns=MEDICATION_PATTERNS):
    """Binary medication flags from prescriptions on strictly prior admissions."""
    prescriptions = read_filtered_chunks(
        f"{mimic_path}/prescriptions.csv.gz",
        subject_ids=master_df["subject_id"].unique(),
        columns=["subject_id", "hadm_id", "drug"],
    )

    out = master_df.copy()
    if prescriptions.empty:
        for med_class in medication_patterns:
            out[med_class] = 0
        return out

    presc_timed = pd.merge(
        prescriptions,
        subject_admissions[["hadm_id", "admittime"]].rename(
            columns={"admittime": "presc_admittime"}
        ),
        on="hadm_id",
        how="inner",
    )

    for med_class, pattern in medication_patterns.items():
        matched = presc_timed[
            presc_timed["drug"].str.contains(pattern, case=False, na=False, regex=True)
        ]
        events = matched[["subject_id", "presc_admittime"]].drop_duplicates()
        out[med_class] = _flag_prior_events(out, events, "presc_admittime", strict=True)
    return out


def finalize(master_df):
    """Encode gender, rename lab columns, drop identifiers used only for timing.

    subject_id survives: it is the group key for the patient-level split and is
    dropped just before training.
    """
    out = master_df.copy()
    out["gender"] = (out["gender"] == "M").astype(int)
    out = out.rename(columns=LAB_RENAME)

    for column in LAB_RENAME.values():
        if column not in out.columns:
            out[column] = pd.NA

    return out.drop(columns=["hadm_id", "admittime_index"])
