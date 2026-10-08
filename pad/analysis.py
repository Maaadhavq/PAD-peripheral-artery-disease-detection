"""Analyses that go beyond a single headline number.

One AUC on one split says very little: it has no uncertainty attached, it hides
whichever subgroup the model serves worst, and it does not say which features
earned it. Each function here answers one of those.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from pad.config import (
    AGE_BINS,
    AGE_LABELS,
    COMORBIDITY_CODES,
    LAB_RENAME,
    MEDICATION_PATTERNS,
    RANDOM_SEED,
)
from pad.train import build_models, cross_validate, fit_pipeline, make_pipeline

FEATURE_GROUPS = {
    "demographics": ["gender", "age_at_admission"],
    "labs": list(LAB_RENAME.values()),
    "comorbidities": list(COMORBIDITY_CODES.keys()),
    "medications": list(MEDICATION_PATTERNS.keys()),
}


def bootstrap_auc(y_true, y_prob, n_resamples=1000, alpha=0.05, seed=RANDOM_SEED):
    """Percentile confidence interval for AUC by resampling the test set.

    The point estimate is one draw from a sampling distribution; this shows how
    wide that distribution is. Resamples that end up single-class are skipped,
    since AUC is undefined for them.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))

    if not scores:
        return {"auc": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_resamples": 0}

    scores = np.asarray(scores)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "lo": float(np.percentile(scores, 100 * alpha / 2)),
        "hi": float(np.percentile(scores, 100 * (1 - alpha / 2))),
        "n_resamples": len(scores),
    }


def subgroup_performance(X_test, y_test, y_prob, min_rows=30):
    """AUC per subgroup, by gender and by age bracket.

    A model can look fine overall and fail a subgroup badly. Group sizes are
    reported alongside so small strata are visibly small, and groups below
    ``min_rows`` are marked rather than quietly dropped - "we cannot tell" is a
    different statement from "it performs poorly".
    """
    rows = []

    def add(group_name, label, mask):
        n = int(mask.sum())
        positives = int(np.asarray(y_test)[mask].sum()) if n else 0
        entry = {
            "group": group_name,
            "subgroup": label,
            "n": n,
            "n_positive": positives,
            "auc": np.nan,
            "note": "",
        }
        if n < min_rows:
            entry["note"] = f"too few rows (<{min_rows}) to estimate"
        elif positives == 0 or positives == n:
            entry["note"] = "single class in this subgroup"
        else:
            entry["auc"] = float(roc_auc_score(np.asarray(y_test)[mask],
                                               np.asarray(y_prob)[mask]))
        rows.append(entry)

    gender = X_test["gender"].to_numpy()
    add("gender", "male", gender == 1)
    add("gender", "female", gender == 0)

    brackets = pd.cut(X_test["age_at_admission"], bins=AGE_BINS, labels=AGE_LABELS)
    for label in AGE_LABELS:
        add("age bracket", label, (brackets == label).to_numpy())

    return pd.DataFrame(rows)


def threshold_table(y_true, y_prob, thresholds=None):
    """Sensitivity, specificity, PPV and NPV across decision thresholds.

    The project argues for AUC on the grounds that the threshold is tunable to
    the cost of a missed case. This is the table that makes that argument
    concrete instead of rhetorical.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = thresholds if thresholds is not None else np.arange(0.05, 1.0, 0.05)

    rows = []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": round(float(threshold), 3),
            "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
            "npv": tn / (tn + fn) if (tn + fn) else np.nan,
            "flagged": int(pred.sum()),
        })
    return pd.DataFrame(rows)


def ablation(X_train, y_train, groups_train, model_name="Logistic Regression",
             n_splits=5, seed=RANDOM_SEED):
    """Cross-validated AUC with each feature group removed in turn.

    Measured on the training set by CV, never on the test set - this is a model
    development question, and answering it against the test set would turn the
    held-out estimate into a selection target.
    """
    def cv_auc(columns):
        results = cross_validate(
            X_train[columns], y_train, groups_train, n_splits=n_splits, seed=seed
        )
        return float(results.loc[model_name, "cv_auc_mean"])

    full_columns = list(X_train.columns)
    baseline = cv_auc(full_columns)

    rows = [{"features": "all", "n_features": len(full_columns),
             "cv_auc": baseline, "delta": 0.0}]

    for group, columns in FEATURE_GROUPS.items():
        remaining = [c for c in full_columns if c not in columns]
        if not remaining:
            continue
        score = cv_auc(remaining)
        rows.append({
            "features": f"without {group}",
            "n_features": len(remaining),
            "cv_auc": score,
            "delta": score - baseline,
        })

    return pd.DataFrame(rows).sort_values("cv_auc", ascending=False)


def demographics_baseline(X_train, y_train, groups_train, X_test, y_test,
                          model_name="Logistic Regression", seed=RANDOM_SEED):
    """How much the clinical features add over age and gender alone.

    If the full model does not clearly beat this, the clinical data is not
    earning its place and that is the honest headline.
    """
    demographic_columns = FEATURE_GROUPS["demographics"]

    def fit_and_score(columns):
        model = build_models(seed=seed)[model_name]
        pipeline = fit_pipeline(
            make_pipeline(model), X_train[columns], y_train, model_name, seed
        )
        prob = pipeline.predict_proba(X_test[columns])[:, 1]
        return float(roc_auc_score(y_test, prob))

    baseline_auc = fit_and_score(demographic_columns)
    full_auc = fit_and_score(list(X_train.columns))

    return pd.DataFrame([
        {"model": "age + gender only", "n_features": len(demographic_columns),
         "test_auc": baseline_auc},
        {"model": "all features", "n_features": X_train.shape[1],
         "test_auc": full_auc},
        {"model": "difference", "n_features": None,
         "test_auc": full_auc - baseline_auc},
    ])


def classification_summary(y_true, y_pred):
    """Precision, recall and F1 for the PAD class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0
    )
    return {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "support": int(support[0]),
    }
