"""
test_drift_detector.py
-----------------------
Tests pour DriftDetector: PSI, KS, Chi², recommendations.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


# Fixtures

def _make_ref_df(n=500, seed=42) -> pd.DataFrame:
    """DataFrame de référence stable."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "surface_m2" : rng.normal(100, 25, n).clip(20, 500),
        "nb_chambres": rng.integers(1, 6, n).astype(float),
        "age_bien"   : rng.normal(15, 8, n).clip(0, 80),
        "ville"      : rng.choice(["Casablanca", "Rabat", "Marrakech"], n),
    })


def _make_similar_df(n=200, seed=99) -> pd.DataFrame:
    """Données similaires à la référence (pas de drift)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "surface_m2" : rng.normal(102, 26, n).clip(20, 500),   # légèrement différent
        "nb_chambres": rng.integers(1, 6, n).astype(float),
        "age_bien"   : rng.normal(15.5, 8, n).clip(0, 80),
        "ville"      : rng.choice(["Casablanca", "Rabat", "Marrakech"], n),
    })


def _make_drifted_df(n=200, seed=77) -> pd.DataFrame:
    """Données très différentes (drift fort)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "surface_m2" : rng.normal(250, 60, n).clip(20, 500),   # ↑ distribution totalement différente
        "nb_chambres": rng.integers(4, 7, n).astype(float),    # ↑ beaucoup plus de chambres
        "age_bien"   : rng.normal(2, 1, n).clip(0, 80),        # ↑ biens très récents
        "ville"      : rng.choice(["Agadir", "Tanger"], n),    # ↑ nouvelles villes
    })


@pytest.fixture
def detector():
    from drift_detector import DriftDetector
    return DriftDetector(
        reference_data=_make_ref_df(),
        output_dir="/tmp/test_drift_reports",
    )


# Tests PSI

class TestPSI:
    def test_psi_stable_same_distribution(self, detector):
        report = detector.detect(_make_similar_df(), save_report=False)
        assert report.dataset_psi < 0.15  # légèrement différent mais stable

    def test_psi_high_different_distribution(self, detector):
        report = detector.detect(_make_drifted_df(), save_report=False)
        assert report.dataset_psi > 0.10  # drift détecté

    def test_psi_identical_data_is_zero(self):
        from drift_detector import _compute_psi
        data = np.random.normal(100, 20, 500)
        psi = _compute_psi(data, data)
        assert psi < 0.01  # identique → PSI ≈ 0

    def test_psi_very_different_is_high(self):
        from drift_detector import _compute_psi
        ref = np.random.normal(0, 1, 500)
        cur = np.random.normal(10, 1, 500)   # distribution totalement décalée
        psi = _compute_psi(ref, cur)
        assert psi > 0.20


# Tests Features

class TestFeatureDrift:
    def test_all_features_checked(self, detector):
        report = detector.detect(_make_similar_df(), save_report=False)
        feature_names = {r.feature for r in report.feature_results}
        assert "surface_m2" in feature_names
        assert "nb_chambres" in feature_names
        assert "age_bien" in feature_names
        assert "ville" in feature_names

    def test_numerical_uses_psi(self, detector):
        report = detector.detect(_make_similar_df(), save_report=False)
        num_results = [r for r in report.feature_results if r.test_used == "psi"]
        assert len(num_results) >= 3  # surface_m2, nb_chambres, age_bien

    def test_categorical_uses_chi2(self, detector):
        report = detector.detect(_make_similar_df(), save_report=False)
        cat_results = [r for r in report.feature_results if r.test_used == "chi2"]
        assert len(cat_results) >= 1  # ville

    def test_drifted_data_detects_more_features(self, detector):
        report_stable = detector.detect(_make_similar_df(), save_report=False)
        report_drift  = detector.detect(_make_drifted_df(), save_report=False)
        assert report_drift.n_drifted >= report_stable.n_drifted

    def test_feature_result_has_statistics(self, detector):
        report = detector.detect(_make_similar_df(), save_report=False)
        for r in report.feature_results:
            assert r.statistic >= 0
            assert r.severity in ("stable", "warning", "drift")


# Tests Predictions Drift

class TestPredictionDrift:
    def test_no_drift_similar_predictions(self, detector):
        rng = np.random.default_rng(0)
        preds_ref = rng.normal(500_000, 100_000, 500)
        preds_cur = rng.normal(505_000, 102_000, 200)  # très similaire

        report = detector.detect(
            _make_similar_df(),
            predictions_ref=preds_ref,
            predictions_cur=preds_cur,
            save_report=False,
        )
        assert report.prediction_drift is not None
        assert report.prediction_drift.psi < 0.30

    def test_drift_different_predictions(self, detector):
        rng = np.random.default_rng(0)
        preds_ref = rng.normal(500_000, 50_000, 500)
        preds_cur = rng.normal(1_500_000, 200_000, 200)  # doublement des prix

        report = detector.detect(
            _make_similar_df(),
            predictions_ref=preds_ref,
            predictions_cur=preds_cur,
            save_report=False,
        )
        assert report.prediction_drift is not None
        pred_drift = report.prediction_drift
        assert abs(pred_drift.mean_shift_pct) > 50  # shift > 50%


# Tests Recommandations

class TestRecommendations:
    def test_ok_when_stable(self, detector):
        report = detector.detect(_make_similar_df(), save_report=False)
        assert report.recommendation in ("ok", "monitor")  # données similaires

    def test_retrain_when_strong_drift(self, detector):
        report = detector.detect(_make_drifted_df(), save_report=False)
        # avec un drift aussi fort, doit recommander monitor ou retrain
        assert report.recommendation in ("monitor", "retrain")

    def test_retrain_when_prediction_drift(self, detector):
        from drift_detector import DriftDetector
        rng = np.random.default_rng(0)
        preds_ref = rng.normal(500_000, 50_000, 500)
        preds_cur = rng.normal(2_000_000, 300_000, 200)  # shift massif

        report = detector.detect(
            _make_similar_df(),
            predictions_ref=preds_ref,
            predictions_cur=preds_cur,
            save_report=False,
        )
        assert report.recommendation in ("monitor", "retrain")

    def test_recommendation_in_valid_values(self, detector):
        report = detector.detect(_make_similar_df(), save_report=False)
        assert report.recommendation in ("ok", "monitor", "retrain")


# Tests DriftReport

class TestDriftReport:
    def test_report_has_required_fields(self, detector):
        report = detector.detect(_make_similar_df(), save_report=False)
        assert report.timestamp is not None
        assert report.n_features_checked > 0
        assert isinstance(report.n_drifted, int)
        assert isinstance(report.dataset_psi, float)
        assert report.ref_size == 500
        assert report.cur_size == 200

    def test_report_to_dict_serializable(self, detector):
        import json
        report = detector.detect(_make_similar_df(), save_report=False)
        d = report.to_dict()
        json_str = json.dumps(d)  # doit être sérialisable
        assert len(json_str) > 100

    def test_drifted_features_list(self, detector):
        report = detector.detect(_make_drifted_df(), save_report=False)
        drifted = report.drifted_features()
        assert isinstance(drifted, list)

    def test_has_drift_property(self, detector):
        report_stable = detector.detect(_make_similar_df(), save_report=False)
        report_drift  = detector.detect(_make_drifted_df(), save_report=False)
        # Le drift fort doit avoir plus de features driftées
        assert report_drift.n_drifted >= report_stable.n_drifted

    def test_summary_returns_string(self, detector):
        report = detector.detect(_make_similar_df(), save_report=False)
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 10


# Tests Update Reference

class TestUpdateReference:
    def test_update_reference_changes_ref_size(self):
        from drift_detector import DriftDetector
        ref = _make_ref_df(n=500)
        det = DriftDetector(reference_data=ref)
        assert len(det.reference) == 500

        new_ref = _make_ref_df(n=1000, seed=55)
        det.update_reference(new_ref)
        assert len(det.reference) == 1000

    def test_trend_none_with_one_report(self, detector):
        detector.detect(_make_similar_df(), save_report=False)
        # Un seul rapport → pas encore de tendance
        assert detector.trend() is None

    def test_trend_available_with_two_reports(self, detector):
        detector.detect(_make_similar_df(), save_report=False)
        detector.detect(_make_drifted_df(), save_report=False)
        trend = detector.trend()
        assert trend is not None
        assert "psi_history" in trend
        assert len(trend["psi_history"]) == 2
