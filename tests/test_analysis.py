"""Calibration and the analyses that go beyond a single AUC."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pad import analysis as analysis_mod
from pad import calibrate as cal_mod
from pad import train as train_mod
from pad.config import FEATURE_COLUMNS
from pad.explain import top_factors, unwrap_pipeline


@pytest.fixture(scope="module")
def splits(dataset):
    X, y, groups = train_mod.split_xy(dataset)
    X_train, X_test, y_train, y_test, groups_train = train_mod.patient_split(
        X, y, groups
    )
    X_fit, X_cal, y_fit, y_cal, groups_fit = cal_mod.split_calibration(
        X_train, y_train, groups_train
    )
    return {
        "X": X, "y": y, "groups": groups,
        "X_fit": X_fit, "y_fit": y_fit, "groups_fit": groups_fit,
        "X_cal": X_cal, "y_cal": y_cal,
        "X_test": X_test, "y_test": y_test,
    }


@pytest.fixture(scope="module")
def fitted(splits):
    model = train_mod.build_models()["Logistic Regression"]
    pipeline = train_mod.fit_pipeline(
        train_mod.make_pipeline(model), splits["X_fit"], splits["y_fit"]
    )
    calibrated, method = cal_mod.calibrate(pipeline, splits["X_cal"], splits["y_cal"])
    return {"pipeline": pipeline, "calibrated": calibrated, "method": method}


class TestCalibrationSplit:
    def test_fit_and_calibration_share_no_patient(self, splits):
        fit_patients = set(splits["groups"].loc[splits["X_fit"].index])
        cal_patients = set(splits["groups"].loc[splits["X_cal"].index])
        assert not (fit_patients & cal_patients)

    def test_calibration_never_overlaps_test(self, splits):
        cal_patients = set(splits["groups"].loc[splits["X_cal"].index])
        test_patients = set(splits["groups"].loc[splits["X_test"].index])
        assert not (cal_patients & test_patients)

    def test_method_falls_back_to_platt_when_small(self):
        assert cal_mod.choose_method(50) == "sigmoid"
        assert cal_mod.choose_method(5000) == "isotonic"


class TestCalibration:
    def test_ranking_is_preserved(self, splits, fitted):
        """Calibration is a monotone remap, so AUC should barely move. A large
        change means the calibrator is doing something it should not."""
        comparison = cal_mod.compare(
            splits["y_test"],
            fitted["pipeline"].predict_proba(splits["X_test"])[:, 1],
            fitted["calibrated"].predict_proba(splits["X_test"])[:, 1],
        )
        assert abs(comparison["before"]["auc"] - comparison["after"]["auc"]) < 0.05

    def test_outputs_stay_probabilities(self, splits, fitted):
        prob = fitted["calibrated"].predict_proba(splits["X_test"])[:, 1]
        assert prob.min() >= 0.0
        assert prob.max() <= 1.0

    def test_reliability_curve_drops_empty_bins(self):
        curve = cal_mod.reliability_curve([0, 1, 0, 1], [0.01, 0.99, 0.02, 0.98])
        assert all(count > 0 for count in curve["count"])
        assert len(curve["bin_centre"]) == len(curve["observed"])

    def test_perfect_predictions_are_perfectly_calibrated(self):
        y = [0, 0, 1, 1]
        metrics = cal_mod.calibration_metrics(y, [0.0, 0.0, 1.0, 1.0])
        assert metrics["brier"] == pytest.approx(0.0)
        assert metrics["ece"] == pytest.approx(0.0)


class TestShapThroughCalibrator:
    def test_unwrap_reaches_the_pipeline(self, fitted):
        """CalibratedClassifierCV is not subscriptable, so SHAP cannot reach the
        preprocessing steps through it without unwrapping first."""
        assert isinstance(unwrap_pipeline(fitted["calibrated"]), Pipeline)

    def test_unwrap_is_a_no_op_for_a_plain_pipeline(self, fitted):
        assert unwrap_pipeline(fitted["pipeline"]) is fitted["pipeline"]

    def test_factors_can_be_computed_on_a_calibrated_model(self, splits, fitted):
        bundle = {"pipeline": fitted["calibrated"], "model_name": "LR",
                  "features": FEATURE_COLUMNS}
        factors = top_factors(bundle, splits["X_test"].iloc[0].to_dict(), k=3,
                              background=splits["X_fit"].head(50))
        assert len(factors) == 3
        assert all(f["label"] for f in factors)


class TestBootstrap:
    def test_interval_contains_the_point_estimate(self, splits, fitted):
        prob = fitted["calibrated"].predict_proba(splits["X_test"])[:, 1]
        ci = analysis_mod.bootstrap_auc(splits["y_test"], prob, n_resamples=200)
        assert ci["lo"] <= ci["auc"] <= ci["hi"]
        assert ci["n_resamples"] > 0

    def test_is_deterministic_for_a_fixed_seed(self, splits, fitted):
        prob = fitted["calibrated"].predict_proba(splits["X_test"])[:, 1]
        first = analysis_mod.bootstrap_auc(splits["y_test"], prob, n_resamples=100)
        second = analysis_mod.bootstrap_auc(splits["y_test"], prob, n_resamples=100)
        assert first == second

    def test_single_class_resamples_do_not_crash(self):
        result = analysis_mod.bootstrap_auc([1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4],
                                            n_resamples=50)
        assert result["n_resamples"] == 0
        assert np.isnan(result["auc"])


class TestSubgroups:
    def test_small_subgroups_are_marked_not_scored(self, splits):
        """"We cannot tell" must not be rendered as a number that reads as
        evidence."""
        prob = np.linspace(0, 1, len(splits["y_test"]))
        table = analysis_mod.subgroup_performance(
            splits["X_test"], splits["y_test"], prob, min_rows=10_000
        )
        assert table["auc"].isna().all()
        assert table["note"].str.contains("too few rows").all()

    def test_group_sizes_are_reported(self, splits, fitted):
        prob = fitted["calibrated"].predict_proba(splits["X_test"])[:, 1]
        table = analysis_mod.subgroup_performance(
            splits["X_test"], splits["y_test"], prob
        )
        assert (table["n"] >= 0).all()
        assert table["n"].sum() >= len(splits["y_test"])  # gender + age both counted

    def test_covers_both_genders_and_every_age_bracket(self, splits, fitted):
        prob = fitted["calibrated"].predict_proba(splits["X_test"])[:, 1]
        table = analysis_mod.subgroup_performance(
            splits["X_test"], splits["y_test"], prob
        )
        assert set(table[table["group"] == "gender"]["subgroup"]) == {"male", "female"}
        assert len(table[table["group"] == "age bracket"]) == 6


class TestThresholds:
    def test_monotone_in_the_expected_directions(self):
        y_true = np.array([0] * 50 + [1] * 50)
        prob = np.concatenate([np.linspace(0, 0.6, 50), np.linspace(0.4, 1, 50)])
        table = analysis_mod.threshold_table(y_true, prob)
        assert table["sensitivity"].is_monotonic_decreasing
        assert table["specificity"].is_monotonic_increasing

    def test_flagged_count_falls_as_the_threshold_rises(self, splits, fitted):
        prob = fitted["calibrated"].predict_proba(splits["X_test"])[:, 1]
        table = analysis_mod.threshold_table(splits["y_test"], prob)
        assert table["flagged"].is_monotonic_decreasing


class TestAblation:
    def test_covers_every_feature_group(self, splits):
        table = analysis_mod.ablation(
            splits["X_fit"], splits["y_fit"], splits["groups_fit"], n_splits=3
        )
        rows = set(table["features"])
        assert "all" in rows
        for group in analysis_mod.FEATURE_GROUPS:
            assert f"without {group}" in rows

    def test_feature_groups_cover_every_model_feature(self):
        covered = {c for cols in analysis_mod.FEATURE_GROUPS.values() for c in cols}
        assert covered == set(FEATURE_COLUMNS)

    def test_baseline_reports_the_difference(self, splits):
        table = analysis_mod.demographics_baseline(
            splits["X_fit"], splits["y_fit"], splits["groups_fit"],
            splits["X_test"], splits["y_test"],
        )
        assert list(table["model"])[-1] == "difference"
        assert len(table) == 3
