"""Cohort construction: PAD cases and demographically matched controls."""

import random

import pandas as pd

from pad.config import AGE_BINS, AGE_LABELS, PAD_CODES, RANDOM_SEED


def _matches_codes(diagnoses, codes):
    """Rows whose ICD code starts with any code in the version-matched sets."""
    icd10 = diagnoses["icd_code"].str.startswith(tuple(codes["icd10"]), na=False)
    icd9 = diagnoses["icd_code"].str.startswith(tuple(codes["icd9"]), na=False)
    return (icd10 & (diagnoses["icd_version"] == 10)) | (icd9 & (diagnoses["icd_version"] == 9))


def admission_demographics(admissions, patients):
    """Age at admission and gender for every admission, plus an age bracket."""
    demo = pd.merge(
        admissions[["hadm_id", "subject_id", "admittime"]],
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="left",
    )
    demo["admittime"] = pd.to_datetime(demo["admittime"])
    demo["age_at_adm"] = demo["anchor_age"] + (demo["admittime"].dt.year - demo["anchor_year"])
    demo["age_bracket"] = pd.cut(demo["age_at_adm"], bins=AGE_BINS, labels=AGE_LABELS)
    return demo


def build_cohort(admissions, patients, diagnoses, seed=RANDOM_SEED):
    """Build a 1:1 age/gender matched case-control cohort of admissions.

    Cases are admissions carrying a PAD code. Controls are drawn from patients
    with no PAD code in *any* admission — matching on admissions alone would
    let a PAD patient's other stays become controls, putting the same person on
    both sides of the label. Each control patient contributes a single
    admission so no one is over-represented.
    """
    pad_diagnoses = diagnoses[_matches_codes(diagnoses, PAD_CODES)]
    pad_admission_ids = set(pad_diagnoses["hadm_id"].unique())
    pad_subject_ids = set(pad_diagnoses["subject_id"].unique())

    demo = admission_demographics(admissions, patients)
    pad_demo = demo[demo["hadm_id"].isin(pad_admission_ids)]
    control_pool = demo[~demo["subject_id"].isin(pad_subject_ids)]
    # One admission per control patient, chosen deterministically.
    control_pool = control_pool.sort_values("hadm_id").drop_duplicates("subject_id")

    rng = random.Random(seed)
    control_ids = []
    for (gender, bracket), cases in pad_demo.groupby(["gender", "age_bracket"], observed=True):
        pool = control_pool[
            (control_pool["gender"] == gender) & (control_pool["age_bracket"] == bracket)
        ]["hadm_id"].tolist()
        n = min(len(cases), len(pool))
        control_ids.extend(rng.sample(pool, n))

    cases_df = pd.DataFrame({"hadm_id": sorted(pad_admission_ids), "had_pad": 1})
    controls_df = pd.DataFrame({"hadm_id": control_ids, "had_pad": 0})
    cohort = (
        pd.concat([cases_df, controls_df])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    return cohort


def add_demographics(cohort, admissions, patients):
    """Attach subject_id, gender, age at admission, and the index timestamp.

    admittime is kept as ``admittime_index``: comorbidity and medication
    features are restricted to events at or before it, and it is dropped before
    modelling.
    """
    df = pd.merge(cohort, admissions[["hadm_id", "subject_id", "admittime"]],
                  on="hadm_id", how="left")
    df = pd.merge(df, patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
                  on="subject_id", how="left")
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["age_at_admission"] = df["anchor_age"] + (df["admittime"].dt.year - df["anchor_year"])
    df = df.drop(columns=["anchor_age", "anchor_year"])
    df = df.rename(columns={"admittime": "admittime_index"})
    df["age_at_admission"] = df["age_at_admission"].fillna(df["age_at_admission"].median())
    return df


def subject_admission_times(admissions, subject_ids):
    """Timestamped view of every admission belonging to our subjects.

    Used to date diagnoses and prescriptions, which carry hadm_id but no time.
    """
    subset = admissions[admissions["subject_id"].isin(set(subject_ids))]
    out = subset[["subject_id", "hadm_id", "admittime"]].copy()
    out["admittime"] = pd.to_datetime(out["admittime"])
    return out
