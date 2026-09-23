"""Loading MIMIC-IV tables, including chunked reads for the multi-GB files."""

from pathlib import Path

import pandas as pd

SMALL_TABLES = ["admissions", "patients", "diagnoses_icd", "d_labitems"]
CHUNK_SIZE = 1_000_000


def resolve_mimic_path(path=None):
    """Return the directory holding the MIMIC-IV csv.gz files.

    On Colab this mounts Drive and points at My Drive/mimic_data. Locally it
    falls back to ./mimic_data, so the same notebook runs in both places.
    """
    if path is not None:
        return Path(path)
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        return Path("/content/drive/MyDrive/mimic_data")
    except ImportError:
        return Path("./mimic_data")


def load_tables(mimic_path, tables=None):
    """Read the small MIMIC-IV tables into a dict of DataFrames."""
    mimic_path = Path(mimic_path)
    tables = tables or SMALL_TABLES
    out = {}
    for name in tables:
        out[name] = pd.read_csv(mimic_path / f"{name}.csv.gz", compression="gzip")
    return out


def read_filtered_chunks(filepath, subject_ids, columns=None, itemids=None,
                         chunk_size=CHUNK_SIZE):
    """Stream a large table, keeping only rows for our cohort.

    labevents and prescriptions are far too big to hold in memory, so each
    chunk is filtered down to the subjects (and optionally itemids) we care
    about before anything is retained.
    """
    subject_ids = set(subject_ids)
    itemids = set(itemids) if itemids is not None else None

    kept = []
    for chunk in pd.read_csv(filepath, compression="gzip", chunksize=chunk_size,
                             low_memory=False):
        mask = chunk["subject_id"].isin(subject_ids)
        if itemids is not None:
            mask &= chunk["itemid"].isin(itemids)
        filtered = chunk[mask]
        if columns is not None:
            filtered = filtered[columns]
        if not filtered.empty:
            kept.append(filtered)

    if not kept:
        return pd.DataFrame(columns=columns)
    return pd.concat(kept, ignore_index=True)
