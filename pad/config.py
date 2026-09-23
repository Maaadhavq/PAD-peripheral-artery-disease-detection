"""Clinical code sets, lab definitions and drug patterns used across the pipeline."""

# ICD codes that mark an admission as a PAD case.
PAD_CODES = {
    "icd9": ["4439", "44389", "44381", "44022", "44029", "4408", "4409"],
    "icd10": ["I739", "I7021", "I7022", "I7029", "I708", "I709", "I7391", "I7399"],
}

COMORBIDITY_CODES = {
    "has_diabetes": {
        "icd9": ["250"],
        "icd10": ["E10", "E11", "E13"],
    },
    "has_hypertension": {
        "icd9": ["401", "402", "403", "404", "405"],
        "icd10": ["I10", "I11", "I12", "I13", "I15"],
    },
    "has_heart_disease": {
        "icd9": ["410", "411", "412", "413", "414"],
        "icd10": ["I20", "I21", "I22", "I23", "I24", "I25"],
    },
    "has_stroke_history": {
        "icd9": ["430", "431", "432", "433", "434", "436"],
        "icd10": ["I60", "I61", "I62", "I63", "I64"],
    },
}

# d_labitems labels to pull from labevents.
LAB_TESTS = ["Cholesterol, Total", "Glucose", "Creatinine", "Hemoglobin", "Platelet Count"]

LAB_RENAME = {
    "Cholesterol, Total": "cholesterol",
    "Glucose": "glucose",
    "Creatinine": "creatinine",
    "Hemoglobin": "hemoglobin",
    "Platelet Count": "platelet_count",
}

# Explicit drug names, not substrings: a bare "statin" pattern also matches
# nystatin, an antifungal with nothing to do with cardiovascular risk.
MEDICATION_PATTERNS = {
    "is_on_statin": (
        r"atorvastatin|rosuvastatin|simvastatin|pravastatin"
        r"|lovastatin|pitavastatin|fluvastatin"
    ),
    "is_on_antiplatelet": r"aspirin|clopidogrel|plavix|ticagrelor|prasugrel|brilinta|effient",
}

AGE_BINS = [0, 40, 50, 60, 70, 80, 200]
AGE_LABELS = ["<40", "40-50", "50-60", "60-70", "70-80", "80+"]

# Column order the model sees. Anything not listed here is an identifier or a label.
FEATURE_COLUMNS = (
    ["gender", "age_at_admission"]
    + list(LAB_RENAME.values())
    + list(COMORBIDITY_CODES.keys())
    + list(MEDICATION_PATTERNS.keys())
)

TARGET_COLUMN = "had_pad"
GROUP_COLUMN = "subject_id"
RANDOM_SEED = 42
