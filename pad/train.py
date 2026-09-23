"""Model training: patient-level splits, grouped CV, artifact persistence."""

# lightgbm is imported first on purpose: importing it after scikit-learn
# crashes it on Windows (the two ship clashing OpenMP runtimes).
import lightgbm as lgb  # noqa: F401,E402  (must load before sklearn)

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample
from xgboost import XGBClassifier

from pad.config import FEATURE_COLUMNS, GROUP_COLUMN, RANDOM_SEED, TARGET_COLUMN

ARTIFACT_DIR = Path("artifacts")
MODEL_FILE = "model.joblib"
CARD_FILE = "model_card.json"


def split_xy(df, feature_columns=FEATURE_COLUMNS):
    """Split a finalized dataset into X, y and the patient group key."""
    X = df[list(feature_columns)]
    y = df[TARGET_COLUMN]
    groups = df[GROUP_COLUMN]
    return X, y, groups


def patient_split(X, y, groups, test_size=0.2, seed=RANDOM_SEED):
    """Hold out a test set by patient, so no patient spans both sides.

    A patient with several admissions would otherwise let the model memorise
    their physiology in training and be rewarded for it at test time.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        groups.iloc[train_idx],
    )


def _oversample_indices(y, seed=RANDOM_SEED):
    """Index array that balances the classes by resampling the minority one."""
    y = np.asarray(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    minority, majority = (pos_idx, neg_idx) if len(pos_idx) < len(neg_idx) else (neg_idx, pos_idx)
    oversampled = resample(minority, replace=True, n_samples=len(majority), random_state=seed)
    idx = np.concatenate([majority, oversampled])
    np.random.default_rng(seed).shuffle(idx)
    return idx


def build_models(scale_pos_weight=1.0, seed=RANDOM_SEED):
    """The six classifiers being compared, each with class imbalance handled."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", random_state=seed, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=seed),
        # MLPClassifier supports neither class_weight nor sample_weight, so
        # fit_pipeline oversamples the minority class for it instead.
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=seed
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", scale_pos_weight=scale_pos_weight, random_state=seed
        ),
        "LightGBM": lgb.LGBMClassifier(
            class_weight="balanced", random_state=seed, verbose=-1
        ),
    }


def make_pipeline(model):
    """Imputation and scaling live inside the model, so they never see test data."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def fit_pipeline(pipeline, X, y, model_name="", seed=RANDOM_SEED):
    """Fit a pipeline, oversampling first for the MLP which has no class weights."""
    if "MLP" in model_name:
        idx = _oversample_indices(y, seed=seed)
        return pipeline.fit(X.iloc[idx], y.iloc[idx])
    return pipeline.fit(X, y)


def cross_validate(X_train, y_train, groups_train, n_splits=5, seed=RANDOM_SEED):
    """Grouped K-fold CV on the training set - this is what picks the winner.

    Selecting a model on the held-out test set and then reporting that same
    set's score inflates the result. The test set is touched once, after the
    winner has been chosen here.
    """
    n_splits = min(n_splits, groups_train.nunique())
    cv = GroupKFold(n_splits=n_splits)

    scores = {name: [] for name in build_models()}
    for train_idx, val_idx in cv.split(X_train, y_train, groups=groups_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        fold_spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        for name, model in build_models(fold_spw, seed).items():
            pipeline = fit_pipeline(make_pipeline(model), X_tr, y_tr, name, seed)
            proba = pipeline.predict_proba(X_val)[:, 1]
            scores[name].append(roc_auc_score(y_val, proba))

    results = pd.DataFrame({
        "cv_auc_mean": {k: float(np.mean(v)) for k, v in scores.items()},
        "cv_auc_std": {k: float(np.std(v)) for k, v in scores.items()},
        "n_folds": {k: len(v) for k, v in scores.items()},
    })
    return results.sort_values("cv_auc_mean", ascending=False)


def fit_best(cv_results, X_train, y_train, seed=RANDOM_SEED):
    """Refit the best-scoring CV model on the whole training set."""
    best_name = cv_results["cv_auc_mean"].idxmax()
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = build_models(scale_pos_weight, seed)[best_name]
    pipeline = fit_pipeline(make_pipeline(model), X_train, y_train, best_name, seed)
    return best_name, pipeline


def evaluate(pipeline, X_test, y_test):
    """Held-out metrics for the chosen model. Run once, at the very end."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = pipeline.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)
    return {
        "test_auc": float(roc_auc_score(y_test, proba)),
        "test_accuracy": float(report["accuracy"]),
        "test_recall_pad": float(report["1"]["recall"]),
        "test_precision_pad": float(report["1"]["precision"]),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "n_test": int(len(y_test)),
    }


def roc_points(pipeline, X_test, y_test):
    """False/true positive rates for plotting the ROC curve."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    return fpr, tpr


def save_artifacts(pipeline, model_name, cv_results, test_metrics, cohort_stats,
                   artifact_dir=ARTIFACT_DIR, feature_columns=FEATURE_COLUMNS):
    """Persist the fitted pipeline and a model card describing how it was built.

    The copilot loads these instead of re-running the notebook.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"pipeline": pipeline, "model_name": model_name, "features": list(feature_columns)},
        artifact_dir / MODEL_FILE,
    )

    card = {
        "model_name": model_name,
        "created": date.today().isoformat(),
        "features": list(feature_columns),
        "cv_results": cv_results.to_dict(orient="index"),
        "test_metrics": test_metrics,
        "cohort": cohort_stats,
    }
    (artifact_dir / CARD_FILE).write_text(json.dumps(card, indent=2), encoding="utf-8")
    return artifact_dir / MODEL_FILE, artifact_dir / CARD_FILE


def load_artifacts(artifact_dir=ARTIFACT_DIR):
    """Load the fitted pipeline and model card. Raises if training has not run."""
    artifact_dir = Path(artifact_dir)
    model_path = artifact_dir / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run pad_model.ipynb to train and save one."
        )
    bundle = joblib.load(model_path)
    card_path = artifact_dir / CARD_FILE
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
    return bundle, card
