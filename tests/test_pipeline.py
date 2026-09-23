"""Tests for the leakage guarantees the pipeline is supposed to provide."""

import re

import pandas as pd
import pytest

from pad import cohort as cohort_mod
from pad import features as features_mod
from pad import train as train_mod
from pad.config import FEATURE_COLUMNS, MEDICATION_PATTERNS


class TestCohort:
    def test_no_patient_is_both_case_and_control(self, built_cohort):
        labels_per_patient = built_cohort.groupby("subject_id")["had_pad"].nunique()
        assert (labels_per_patient > 1).sum() == 0

    def test_controls_contribute_one_admission_each(self, built_cohort):
        controls = built_cohort[built_cohort["had_pad"] == 0]
        assert controls["subject_id"].is_unique

    def test_controls_are_age_and_gender_matched(self, built_cohort):
        brackets = pd.cut(
            built_cohort["age_at_admission"],
            bins=cohort_mod.AGE_BINS,
            labels=cohort_mod.AGE_LABELS,
        )
        counts = (
            built_cohort.assign(bracket=brackets)
            .groupby(["gender", "bracket", "had_pad"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        # Controls never exceed cases in a stratum; they fall short only when
        # the matching pool runs dry.
        assert (counts.get(0, 0) <= counts.get(1, 0)).all()

    def test_cohort_is_deterministic(self, tables):
        args = (tables["admissions"], tables["patients"], tables["diagnoses_icd"])
        first = cohort_mod.build_cohort(*args)
        second = cohort_mod.build_cohort(*args)
        pd.testing.assert_frame_equal(first, second)


class TestTemporalLeakage:
    def test_comorbidities_ignore_the_index_admission(self, tables, built_cohort):
        """A code first recorded on the index stay must not become a feature.

        ICD codes are assigned at discharge, so they are not known at admission.
        """
        subject_admissions = cohort_mod.subject_admission_times(
            tables["admissions"], built_cohort["subject_id"].unique()
        )
        first_row = built_cohort.iloc[[0]]
        injected = pd.DataFrame([{
            "subject_id": first_row["subject_id"].iloc[0],
            "hadm_id": first_row["hadm_id"].iloc[0],
            "icd_code": "E11",
            "icd_version": 10,
        }])
        # Only the injected code is passed in, so it is the sole possible source
        # of the flag.
        solo = first_row.copy()
        result = features_mod.add_comorbidities(solo, injected, subject_admissions)
        assert result["has_diabetes"].iloc[0] == 0

    def test_medications_ignore_the_index_admission(self, tables, built_cohort):
        subject_admissions = cohort_mod.subject_admission_times(
            tables["admissions"], built_cohort["subject_id"].unique()
        )
        solo = built_cohort.iloc[[0]].copy()
        events = pd.DataFrame([{
            "subject_id": solo["subject_id"].iloc[0],
            "presc_admittime": solo["admittime_index"].iloc[0],
        }])
        flag = features_mod._flag_prior_events(solo, events, "presc_admittime", strict=True)
        assert flag.iloc[0] == 0

    def test_prior_events_are_counted(self, built_cohort):
        solo = built_cohort.iloc[[0]].copy()
        events = pd.DataFrame([{
            "subject_id": solo["subject_id"].iloc[0],
            "diag_admittime": solo["admittime_index"].iloc[0] - pd.Timedelta(days=365),
        }])
        flag = features_mod._flag_prior_events(solo, events, "diag_admittime", strict=True)
        assert flag.iloc[0] == 1


class TestMedicationPatterns:
    def test_nystatin_is_not_a_statin(self):
        pattern = MEDICATION_PATTERNS["is_on_statin"]
        assert not re.search(pattern, "Nystatin Oral Suspension", re.IGNORECASE)
        assert not re.search(pattern, "nystatin", re.IGNORECASE)

    def test_real_statins_still_match(self):
        pattern = MEDICATION_PATTERNS["is_on_statin"]
        for drug in ["Atorvastatin 40mg", "rosuvastatin", "Simvastatin", "PRAVASTATIN SODIUM"]:
            assert re.search(pattern, drug, re.IGNORECASE), drug

    def test_antiplatelets_match(self):
        pattern = MEDICATION_PATTERNS["is_on_antiplatelet"]
        for drug in ["Aspirin 81mg", "Clopidogrel", "Plavix", "Ticagrelor"]:
            assert re.search(pattern, drug, re.IGNORECASE), drug

    def test_nystatin_patient_is_not_flagged(self, nystatin_patient, tables, built_cohort,
                                             tmp_path):
        subject_id, prescriptions = nystatin_patient
        path = tmp_path / "prescriptions.csv.gz"
        prescriptions.to_csv(path, index=False, compression="gzip")

        rows = built_cohort[built_cohort["subject_id"] == subject_id]
        if rows.empty:
            pytest.skip("that patient did not land in the cohort")

        subject_admissions = cohort_mod.subject_admission_times(
            tables["admissions"], built_cohort["subject_id"].unique()
        )
        result = features_mod.add_medications(rows.copy(), tmp_path, subject_admissions)
        assert result["is_on_statin"].sum() == 0


class TestFeatures:
    def test_duplicate_lab_itemids_are_averaged(self, dataset):
        """Two glucose itemids must produce one glucose column, not two."""
        assert list(dataset.columns).count("glucose") == 1

    def test_all_feature_columns_present(self, dataset):
        missing = set(FEATURE_COLUMNS) - set(dataset.columns)
        assert not missing

    def test_identifiers_are_dropped(self, dataset):
        assert "hadm_id" not in dataset.columns
        assert "admittime_index" not in dataset.columns

    def test_gender_is_binary(self, dataset):
        assert set(dataset["gender"].unique()) <= {0, 1}

    def test_no_missing_ages(self, dataset):
        assert dataset["age_at_admission"].isna().sum() == 0


class TestTraining:
    def test_split_keeps_patients_on_one_side(self, dataset):
        X, y, groups = train_mod.split_xy(dataset)
        X_train, X_test, _, _, _ = train_mod.patient_split(X, y, groups)
        train_subjects = set(groups.loc[X_train.index])
        test_subjects = set(groups.loc[X_test.index])
        assert not (train_subjects & test_subjects)

    def test_oversampling_balances_classes(self):
        y = pd.Series([0] * 90 + [1] * 10)
        idx = train_mod._oversample_indices(y)
        balanced = y.iloc[idx]
        assert balanced.sum() == (balanced == 0).sum()

    def test_artifacts_round_trip(self, dataset, tmp_path):
        X, y, groups = train_mod.split_xy(dataset)
        X_train, X_test, y_train, y_test, groups_train = train_mod.patient_split(X, y, groups)

        model = train_mod.build_models()["Logistic Regression"]
        pipeline = train_mod.fit_pipeline(train_mod.make_pipeline(model), X_train, y_train)
        metrics = train_mod.evaluate(pipeline, X_test, y_test)

        cv_results = pd.DataFrame(
            {"cv_auc_mean": {"Logistic Regression": 0.9},
             "cv_auc_std": {"Logistic Regression": 0.01},
             "n_folds": {"Logistic Regression": 5}}
        )
        train_mod.save_artifacts(
            pipeline, "Logistic Regression", cv_results, metrics,
            {"n_cases": int(y.sum())}, artifact_dir=tmp_path,
        )

        bundle, card = train_mod.load_artifacts(artifact_dir=tmp_path)
        assert bundle["features"] == FEATURE_COLUMNS
        assert card["model_name"] == "Logistic Regression"

        proba = bundle["pipeline"].predict_proba(X_test.head(1))[0, 1]
        assert 0.0 <= proba <= 1.0

    def test_load_artifacts_without_training_is_explicit(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Run pad_model.ipynb"):
            train_mod.load_artifacts(artifact_dir=tmp_path / "nope")


class TestExplain:
    def test_top_factors_are_ranked_and_labelled(self, dataset, tmp_path):
        from pad import explain as explain_mod

        X, y, groups = train_mod.split_xy(dataset)
        X_train, X_test, y_train, _, _ = train_mod.patient_split(X, y, groups)
        model = train_mod.build_models()["Random Forest"]
        pipeline = train_mod.fit_pipeline(train_mod.make_pipeline(model), X_train, y_train)
        bundle = {"pipeline": pipeline, "model_name": "Random Forest",
                  "features": FEATURE_COLUMNS}

        row = X_test.iloc[0].to_dict()
        factors = explain_mod.top_factors(bundle, row, k=5, background=X_train.head(50))

        assert len(factors) == 5
        magnitudes = [abs(f["shap"]) for f in factors]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert all(f["direction"] in {"increases", "decreases"} for f in factors)
        assert all(f["label"] for f in factors)

    def test_risk_is_a_probability(self, dataset):
        from pad import explain as explain_mod

        X, y, groups = train_mod.split_xy(dataset)
        X_train, X_test, y_train, _, _ = train_mod.patient_split(X, y, groups)
        model = train_mod.build_models()["Logistic Regression"]
        pipeline = train_mod.fit_pipeline(train_mod.make_pipeline(model), X_train, y_train)
        bundle = {"pipeline": pipeline, "model_name": "LR", "features": FEATURE_COLUMNS}

        risk = explain_mod.predict_risk(bundle, X_test.iloc[0].to_dict())
        assert 0.0 <= risk <= 1.0
