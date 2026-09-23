"""SHAP explanations for a single patient's prediction.

The copilot turns these factors into prose; the app plots them. Both need the
same thing: which features pushed this patient's risk up or down, and by how
much.
"""

import numpy as np
import pandas as pd

from pad.config import FEATURE_COLUMNS

# Feature -> how to phrase it in an explanation.
FEATURE_LABELS = {
    "gender": "gender",
    "age_at_admission": "age at admission",
    "cholesterol": "total cholesterol",
    "glucose": "glucose",
    "creatinine": "creatinine",
    "hemoglobin": "hemoglobin",
    "platelet_count": "platelet count",
    "has_diabetes": "diabetes history",
    "has_hypertension": "hypertension history",
    "has_heart_disease": "heart disease history",
    "has_stroke_history": "stroke history",
    "is_on_statin": "statin therapy",
    "is_on_antiplatelet": "antiplatelet therapy",
}


def to_frame(features, feature_columns=FEATURE_COLUMNS):
    """Build a one-row DataFrame in the column order the pipeline expects."""
    if isinstance(features, pd.DataFrame):
        return features[list(feature_columns)]
    return pd.DataFrame([{col: features.get(col) for col in feature_columns}])


def predict_risk(bundle, features):
    """PAD probability for one patient."""
    X = to_frame(features, bundle["features"])
    return float(bundle["pipeline"].predict_proba(X)[0, 1])


def _shap_values(pipeline, X_row, background):
    """SHAP values for the final estimator, computed on transformed inputs.

    The imputer and scaler are part of the pipeline, so the explainer has to see
    what the estimator actually sees, not the raw feature values.
    """
    import shap

    pre = pipeline[:-1]
    model = pipeline[-1]
    X_transformed = pre.transform(X_row)
    background_transformed = pre.transform(background) if background is not None else None

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_transformed)
    except Exception:
        if background_transformed is None:
            background_transformed = np.zeros((1, X_transformed.shape[1]))
        explainer = shap.Explainer(model.predict_proba, background_transformed)
        values = explainer(X_transformed).values

    values = np.asarray(values)
    # Binary classifiers return either (n, features) or (n, features, classes);
    # take the positive class either way.
    if values.ndim == 3:
        values = values[..., -1]
    return values[0]


def top_factors(bundle, features, k=5, background=None):
    """The k features that moved this patient's risk most, largest effect first.

    Returns dicts with the raw value, the SHAP contribution, and whether it
    pushed risk up or down.
    """
    pipeline = bundle["pipeline"]
    columns = bundle["features"]
    X_row = to_frame(features, columns)

    contributions = _shap_values(pipeline, X_row, background)

    factors = []
    for column, shap_value in zip(columns, contributions):
        value = X_row.iloc[0][column]
        factors.append({
            "feature": column,
            "label": FEATURE_LABELS.get(column, column),
            "value": None if pd.isna(value) else float(value),
            "shap": float(shap_value),
            "direction": "increases" if shap_value > 0 else "decreases",
        })

    factors.sort(key=lambda f: abs(f["shap"]), reverse=True)
    return factors[:k]


def factors_to_text(factors):
    """Compact human-readable summary of the factors, for prompts and logs."""
    parts = []
    for factor in factors:
        value = "unknown" if factor["value"] is None else f"{factor['value']:g}"
        parts.append(
            f"{factor['label']} = {value} ({factor['direction']} risk, "
            f"SHAP {factor['shap']:+.3f})"
        )
    return "; ".join(parts)
