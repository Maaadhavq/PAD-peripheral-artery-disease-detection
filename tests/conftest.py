"""Synthetic MIMIC-IV-shaped tables, so the pipeline can be tested without real data.

Nothing here comes from MIMIC-IV. The generator deliberately reproduces the
structures that broke the pipeline before: patients with several admissions,
PAD codes recorded only on the final admission, itemids that share a lab label,
and nystatin prescriptions that a naive "statin" pattern would match.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

N_PATIENTS = 400
PAD_RATE = 0.2
SEED = 7


def _make_tables(seed=SEED, n_patients=N_PATIENTS):
    rng = np.random.default_rng(seed)

    subject_ids = np.arange(10_000, 10_000 + n_patients)
    gender = rng.choice(["M", "F"], n_patients)
    anchor_age = rng.integers(35, 88, n_patients)
    anchor_year = rng.integers(2110, 2200, n_patients)
    is_pad = rng.random(n_patients) < PAD_RATE

    patients = pd.DataFrame({
        "subject_id": subject_ids,
        "gender": gender,
        "anchor_age": anchor_age,
        "anchor_year": anchor_year,
    })

    admission_rows = []
    hadm_id = 500_000
    for subject_id, year in zip(subject_ids, anchor_year):
        for offset in range(rng.integers(1, 4)):
            admission_rows.append((
                hadm_id,
                subject_id,
                f"{year + offset}-{rng.integers(1, 13):02d}-{rng.integers(1, 28):02d} 08:00:00",
            ))
            hadm_id += 1
    admissions = pd.DataFrame(admission_rows, columns=["hadm_id", "subject_id", "admittime"])

    diagnosis_rows = []
    for subject_id, group in admissions.groupby("subject_id"):
        pad_patient = bool(is_pad[subject_id - 10_000])
        hadm_ids = group.sort_values("admittime")["hadm_id"].tolist()
        for h in hadm_ids:
            if rng.random() < (0.6 if pad_patient else 0.25):
                diagnosis_rows.append((subject_id, h, "E11", 10))
            if rng.random() < (0.5 if pad_patient else 0.3):
                diagnosis_rows.append((subject_id, h, "I10", 10))
            diagnosis_rows.append((subject_id, h, "Z999", 10))
        if pad_patient:
            # PAD is coded only on the last admission: earlier stays of the same
            # patient must not be usable as controls.
            diagnosis_rows.append((subject_id, hadm_ids[-1], "I739", 10))
    diagnoses = pd.DataFrame(
        diagnosis_rows, columns=["subject_id", "hadm_id", "icd_code", "icd_version"]
    )

    # Two glucose itemids on purpose: they must be averaged, not duplicated.
    d_labitems = pd.DataFrame({
        "itemid": [50907, 50931, 50809, 50912, 51222, 51265, 99999],
        "label": [
            "Cholesterol, Total", "Glucose", "Glucose", "Creatinine",
            "Hemoglobin", "Platelet Count", "Sodium",
        ],
    })

    lab_rows = []
    for h, subject_id in zip(admissions["hadm_id"], admissions["subject_id"]):
        pad_patient = bool(is_pad[subject_id - 10_000])
        specs = [
            (50907, 190 + 30 * pad_patient, 40), (50931, 120 + 25 * pad_patient, 30),
            (50809, 125 + 25 * pad_patient, 35), (50912, 1.1 + 0.3 * pad_patient, 0.4),
            (51222, 13 - pad_patient, 1.5), (51265, 250, 60), (99999, 140, 3),
        ]
        for itemid, mean, sd in specs:
            if rng.random() < 0.85:
                lab_rows.append((subject_id, h, itemid, float(rng.normal(mean, sd))))
    labevents = pd.DataFrame(
        lab_rows, columns=["subject_id", "hadm_id", "itemid", "valuenum"]
    )

    prescription_rows = []
    for h, subject_id in zip(admissions["hadm_id"], admissions["subject_id"]):
        pad_patient = bool(is_pad[subject_id - 10_000])
        if rng.random() < (0.6 if pad_patient else 0.3):
            prescription_rows.append((subject_id, h, rng.choice(["Atorvastatin", "Simvastatin"])))
        if rng.random() < (0.5 if pad_patient else 0.25):
            prescription_rows.append((subject_id, h, rng.choice(["Aspirin", "Clopidogrel"])))
        prescription_rows.append((subject_id, h, "Acetaminophen"))
    prescriptions = pd.DataFrame(
        prescription_rows, columns=["subject_id", "hadm_id", "drug"]
    )

    return {
        "admissions": admissions,
        "patients": patients,
        "diagnoses_icd": diagnoses,
        "d_labitems": d_labitems,
        "labevents": labevents,
        "prescriptions": prescriptions,
    }


@pytest.fixture(scope="session")
def tables():
    """The synthetic tables as in-memory DataFrames."""
    return _make_tables()


@pytest.fixture(scope="session")
def mimic_path(tables, tmp_path_factory):
    """The same tables written out as csv.gz, for the chunked readers."""
    path = tmp_path_factory.mktemp("mimic_data")
    for name, df in tables.items():
        df.to_csv(Path(path) / f"{name}.csv.gz", index=False, compression="gzip")
    return path


@pytest.fixture(scope="session")
def nystatin_patient(tables):
    """A patient whose only 'statin'-like drug is nystatin, an antifungal.

    Returns (subject_id, prescriptions) with the trap rows added.
    """
    prescriptions = tables["prescriptions"]
    subject_id = int(prescriptions["subject_id"].iloc[0])
    cleaned = prescriptions[
        ~(
            (prescriptions["subject_id"] == subject_id)
            & prescriptions["drug"].str.contains("statin", case=False)
        )
    ]
    trap = pd.DataFrame([
        {"subject_id": subject_id, "hadm_id": h, "drug": "Nystatin Oral Suspension"}
        for h in prescriptions[prescriptions["subject_id"] == subject_id]["hadm_id"].unique()
    ])
    return subject_id, pd.concat([cleaned, trap], ignore_index=True)
