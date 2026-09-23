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


def _is_tree_model(model):
    """True for the tree ensembles TreeExplainer handles directly."""
    return any(
        name in type(model).__name__
        for name in ("RandomForest", "XGB", "LGBM", "GradientBoosting", "DecisionTree",
                     "ExtraTrees", "CatBoost")
    )


def _is_linear_model(model):
    """True for linear models, where SHAP values are exact and cheap."""
    return hasattr(model, "coef_") and hasattr(model, "intercept_")


def _shap_values(pipeline, X_row, background):
    """SHAP values for the final estimator, computed on transformed inputs.

    The imputer and scaler are part of the pipeline, so the explainer has to see
    what the estimator actually sees, not the raw feature values.

    The values come back in whichever output space the chosen explainer uses --
    log-odds for linear and boosted models, probability for a random forest. So
    a magnitude is meaningful for ranking factors *within one explanation*, and
    should not be compared across models.
    """
    import shap

    pre = pipeline[:-1]
    model = pipeline[-1]
    X_transformed = pre.transform(X_row)
    background_transformed = pre.transform(background) if background is not None else None

    # Pick the cheapest explainer the model supports. The permutation fallback
    # is exact but costs seconds per row, which is too slow for an interactive
    # app, so it is the last resort rather than the default.
    if _is_tree_model(model):
        values = shap.TreeExplainer(model).shap_values(X_transformed)
    elif _is_linear_model(model):
        if background_transformed is None:
            background_transformed = np.zeros((1, X_transformed.shape[1]))
        values = shap.LinearExplainer(model, background_transformed).shap_values(X_transformed)
    else:
        if background_transformed is None:
            background_transformed = np.zeros((1, X_transformed.shape[1]))
        explainer = shap.Explainer(model.predict_proba, background_transformed)
        values = explainer(X_transformed, silent=True).values

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
    """Compact human-readable summary of the factors, for prompts and logs.

    The wording is deliberately about the model's score rather than about risk.
    A SHAP value says which way a feature moved *this model's output*, which is
    not a claim about whether the feature causes the disease - a reader (or a
    language model) told that a statin "increases risk" will write something
    clinically false.
    """
    parts = []
    for factor in factors:
        value = "unknown" if factor["value"] is None else f"{factor['value']:g}"
        movement = "pushed the score up" if factor["shap"] > 0 else "pushed the score down"
        parts.append(
            f"{factor['label']} = {value} ({movement}, SHAP {factor['shap']:+.3f})"
        )
    return "; ".join(parts)
