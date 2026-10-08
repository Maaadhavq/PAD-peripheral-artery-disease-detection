"""Probability calibration.

A model that discriminates well can still be badly calibrated: it ranks cases
above controls while its "0.9" means nothing like 90%. That matters here for two
reasons. The cohort is matched 1:1 by construction, so the training base rate is
an artefact rather than a prevalence; and the app shows the number to a reader,
who will read 0.9 as nine-in-ten.

Calibration is fitted on a split held out from training and never seen by the
model, so the calibrator is not learning to correct errors it helped create.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

from pad.config import RANDOM_SEED
from pad.train import patient_split

# Isotonic is non-parametric and more flexible, but it overfits on small
# samples; below this many calibration rows, use sigmoid (Platt) instead.
ISOTONIC_MIN_ROWS = 200


def choose_method(n_rows):
    """Isotonic when there is enough data to support it, otherwise Platt."""
    return "isotonic" if n_rows >= ISOTONIC_MIN_ROWS else "sigmoid"


def split_calibration(X_train, y_train, groups_train, calibration_size=0.25,
                      seed=RANDOM_SEED):
    """Carve a calibration set out of training, split by patient.

    Reuses the same patient-level splitter as the train/test split, so a patient
    cannot appear in both the fitting and the calibration half.
    """
    X_fit, X_cal, y_fit, y_cal, groups_fit = patient_split(
        X_train, y_train, groups_train, test_size=calibration_size, seed=seed
    )
    return X_fit, X_cal, y_fit, y_cal, groups_fit


def calibrate(pipeline, X_cal, y_cal, method=None):
    """Wrap an already-fitted pipeline in a calibrator fitted on held-out data.

    ``cv="prefit"`` tells scikit-learn the underlying model is already trained
    and only the calibration map should be learned here.
    """
    method = method or choose_method(len(y_cal))
    calibrated = CalibratedClassifierCV(pipeline, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated, method


def reliability_curve(y_true, y_prob, n_bins=10):
    """Observed frequency against predicted probability, per bin.

    Returns bin centres, observed rates, mean predicted value and bin counts.
    Empty bins are dropped rather than plotted as zero, which would invent a
    point that no patient supports.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    centres, observed, predicted, counts = [], [], [], []
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            continue
        centres.append((edges[b] + edges[b + 1]) / 2)
        observed.append(float(y_true[mask].mean()))
        predicted.append(float(y_prob[mask].mean()))
        counts.append(count)

    return {
        "bin_centre": centres,
        "observed": observed,
        "predicted": predicted,
        "count": counts,
    }


def calibration_metrics(y_true, y_prob, n_bins=10):
    """Brier score plus expected and maximum calibration error."""
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    counts = np.asarray(curve["count"], dtype=float)
    gaps = np.abs(np.asarray(curve["observed"]) - np.asarray(curve["predicted"]))

    total = counts.sum()
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float((counts * gaps).sum() / total) if total else 0.0,
        "mce": float(gaps.max()) if len(gaps) else 0.0,
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def compare(y_true, prob_before, prob_after, n_bins=10):
    """Calibration metrics before and after, for a side-by-side report.

    AUC is included deliberately: calibration is a monotone remap, so it should
    leave ranking almost untouched. A large AUC change means something is wrong.
    """
    return {
        "before": calibration_metrics(y_true, prob_before, n_bins),
        "after": calibration_metrics(y_true, prob_after, n_bins),
    }
