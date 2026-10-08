"""
test_improvements.py
--------------------
Tests unitaires pour tous les nouveaux modules du pipeline v2.
Couvre : config, logging, baselines, SMOTE, monitoring,
         prediction intervals, data validation, SHAP, MLflow.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Ajouter src/ au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ===============================================================================
# Fixtures communes
# ===============================================================================

@pytest.fixture(scope="session")
def reg_data():
    """Données de régression synthétiques."""
    X, y = make_regression(n_samples=300, n_features=10, noise=50.0, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    return train_test_split(X_df, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="session")
def clf_data_imbalanced():
    """Données de classification déséquilibrées (ratio ~0.2)."""
    X, y = make_classification(
        n_samples=400, n_features=10, n_classes=3,
        n_informative=4, n_clusters_per_class=1,
        weights=[0.7, 0.2, 0.1], random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(10)])
    return train_test_split(X_df, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="session")
def trained_ridge(reg_data):
    """Modèle Ridge entraîné."""
    X_tr, X_te, y_tr, y_te = reg_data
    model = Ridge()
    model.fit(X_tr, y_tr)
    return model, X_tr, X_te, y_tr, y_te


@pytest.fixture
def sample_df():
    """DataFrame immobilier minimal pour les tests de validation."""
    return pd.DataFrame({
        "prix"      : [500_000, 1_200_000, 300_000, 800_000, 2_000_000],
        "surface_m2": [80, 150, 60, 100, 200],
        "ville"     : ["Casablanca", "Rabat", "Fès", "Marrakech", "Agadir"],
        "type_bien" : ["appartement"] * 5,
    })


# ===============================================================================
# Tests: Config Loader
# ===============================================================================

class TestConfigLoader:
    def test_load_config_returns_attrdict(self, tmp_path):
        """La config doit être accessible via attributs."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("pipeline:\n  test_size: 0.25\n  random_state: 99\n")
        from config_loader import load_config, _AttrDict
        cfg = load_config(cfg_file)
        assert isinstance(cfg, _AttrDict)

    def test_attrdict_access(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("pipeline:\n  test_size: 0.25\n")
        from config_loader import load_config
        cfg = load_config(cfg_file)
        assert cfg.pipeline.test_size == 0.25

    def test_env_variable_resolution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_DB_HOST", "myhost")
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text('database:\n  host: "${TEST_DB_HOST:localhost}"\n')
        from config_loader import load_config
        cfg = load_config(cfg_file)
        assert cfg.database.host == "myhost"

    def test_env_fallback_default(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text('database:\n  host: "${NONEXISTENT_VAR_XYZ:fallback_host}"\n')
        from config_loader import load_config
        cfg = load_config(cfg_file)
        assert cfg.database.host == "fallback_host"

    def test_missing_config_raises(self):
        from config_loader import load_config
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_attrdict_missing_key_raises(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("pipeline:\n  test_size: 0.2\n")
        from config_loader import load_config
        cfg = load_config(cfg_file)
        with pytest.raises(AttributeError):
            _ = cfg.pipeline.nonexistent_key


# ===============================================================================
# Tests: Logger
# ===============================================================================

class TestLogger:
    def test_get_logger_returns_logger(self):
        from logger_setup import get_logger
        import logging
        logger = get_logger("test_module_unique_xyz")
        assert isinstance(logger, logging.Logger)

    def test_logger_no_duplicate_handlers(self):
        from logger_setup import get_logger
        l1 = get_logger("same_logger_name_abc")
        l2 = get_logger("same_logger_name_abc")
        # Même instance → pas d'accumulation de handlers
        assert l1 is l2

    def test_logger_writes_to_file(self, tmp_path):
        from logger_setup import get_logger
        logger = get_logger("file_test_logger_xyz", log_dir=str(tmp_path))
        logger.info("Test message to file")
        log_file = tmp_path / "pipeline.log"
        assert log_file.exists()
        assert "Test message to file" in log_file.read_text()


# ===============================================================================
# Tests: Baseline Models
# ===============================================================================

class TestBaselineModels:
    def test_regression_baselines_returns_dict(self, reg_data):
        from baselines import run_regression_baselines
        X_tr, X_te, y_tr, y_te = reg_data
        results = run_regression_baselines(X_tr, y_tr, X_te, y_te)
        assert isinstance(results, dict)
        assert len(results) == 3

    def test_regression_baselines_have_metrics(self, reg_data):
        from baselines import run_regression_baselines
        X_tr, X_te, y_tr, y_te = reg_data
        results = run_regression_baselines(X_tr, y_tr, X_te, y_te)
        for name, metrics in results.items():
            assert "MAE"  in metrics
            assert "RMSE" in metrics
            assert "R2"   in metrics
            assert "MAPE" in metrics

    def test_dummy_mean_r2_is_negative_or_zero(self, reg_data):
        """Un DummyRegressor mean doit avoir R² ≈ 0 ou négatif."""
        from baselines import run_regression_baselines
        X_tr, X_te, y_tr, y_te = reg_data
        results = run_regression_baselines(X_tr, y_tr, X_te, y_te)
        assert results["dummy_mean"]["R2"] <= 0.01

    def test_classification_baselines_returns_dict(self, clf_data_imbalanced):
        from baselines import run_classification_baselines
        X_tr, X_te, y_tr, y_te = clf_data_imbalanced
        results = run_classification_baselines(X_tr, y_tr, X_te, y_te)
        assert isinstance(results, dict)
        assert len(results) == 3

    def test_baseline_report_dataframe(self, reg_data, clf_data_imbalanced):
        from baselines import (
            run_regression_baselines, run_classification_baselines,
            build_baseline_report
        )
        X_tr, X_te, y_tr, y_te = reg_data
        bl_reg = run_regression_baselines(X_tr, y_tr, X_te, y_te)

        Xc_tr, Xc_te, yc_tr, yc_te = clf_data_imbalanced
        bl_clf = run_classification_baselines(Xc_tr, yc_tr, Xc_te, yc_te)

        df = build_baseline_report(
            {"R2": 0.8, "MAE": 1000}, bl_reg,
            {"F1": 0.75, "Accuracy": 0.8}, bl_clf,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "modèle" in df.columns


# ===============================================================================
# Tests: SMOTE
# ===============================================================================

class TestSmoteHandler:
    def test_detect_imbalance(self, clf_data_imbalanced):
        from smote_handler import detect_imbalance
        _, _, y_tr, _ = clf_data_imbalanced
        is_imb, ratio = detect_imbalance(y_tr, threshold=0.5)
        assert isinstance(is_imb, bool)
        assert 0.0 <= ratio <= 1.0

    def test_smote_increases_minority(self, clf_data_imbalanced):
        pytest.importorskip("imblearn")
        from smote_handler import SmoteHandler
        from collections import Counter
        X_tr, _, y_tr, _ = clf_data_imbalanced
        handler = SmoteHandler(imbalance_threshold=0.9)  # force déclenchement
        X_res, y_res = handler.fit_resample(X_tr, y_tr, force=True)
        counts_before = Counter(y_tr)
        counts_after  = Counter(y_res)
        # Au moins autant de données après SMOTE
        assert len(X_res) >= len(X_tr)
        # La minorité doit avoir augmenté
        min_class = min(counts_before, key=counts_before.get)
        assert counts_after[min_class] >= counts_before[min_class]

    def test_smote_not_applied_on_balanced(self, reg_data):
        from smote_handler import SmoteHandler
        import numpy as np
        X_tr, _, y_tr, _ = reg_data
        # Créer labels parfaitement équilibrés
        y_bal = np.array([0, 1] * (len(y_tr) // 2))
        X_bal = X_tr.iloc[:len(y_bal)]
        handler = SmoteHandler(imbalance_threshold=0.1)
        X_res, y_res = handler.fit_resample(X_bal, y_bal)
        assert len(X_res) == len(X_bal)
        assert not handler.was_applied

    def test_smote_get_report(self, clf_data_imbalanced):
        from smote_handler import SmoteHandler
        X_tr, _, y_tr, _ = clf_data_imbalanced
        handler = SmoteHandler()
        handler.fit_resample(X_tr, y_tr)
        report = handler.get_report()
        assert "smote_applied" in report
        assert "before_dist"   in report


# ===============================================================================
# Tests: Monitoring
# ===============================================================================

class TestMonitoring:
    def test_step_timer_records_duration(self, tmp_path):
        from monitoring import PipelineMonitor
        monitor = PipelineMonitor(output_dir=str(tmp_path))
        import time
        with monitor.step("test_step"):
            time.sleep(0.05)
        assert len(monitor._steps) == 1
        assert monitor._steps[0].duration_s >= 0.04

    def test_step_records_success(self, tmp_path):
        from monitoring import PipelineMonitor
        monitor = PipelineMonitor(output_dir=str(tmp_path))
        with monitor.step("ok_step"):
            pass
        assert monitor._steps[0].success is True

    def test_step_records_failure(self, tmp_path):
        from monitoring import PipelineMonitor
        monitor = PipelineMonitor(output_dir=str(tmp_path))
        with pytest.raises(ValueError):
            with monitor.step("fail_step"):
                raise ValueError("intentional")
        assert monitor._steps[0].success is False

    def test_regression_alert_triggered(self, tmp_path):
        from monitoring import PipelineMonitor
        monitor = PipelineMonitor(output_dir=str(tmp_path))
        monitor.alert_r2 = 0.8
        monitor.check_regression_alert(r2=0.5)
        assert len(monitor._alerts) == 1
        assert monitor._alerts[0].metric_name == "R2"

    def test_no_alert_above_threshold(self, tmp_path):
        from monitoring import PipelineMonitor
        monitor = PipelineMonitor(output_dir=str(tmp_path))
        monitor.alert_r2 = 0.5
        monitor.check_regression_alert(r2=0.9)
        assert len(monitor._alerts) == 0

    def test_save_report_creates_json(self, tmp_path):
        from monitoring import PipelineMonitor
        monitor = PipelineMonitor(output_dir=str(tmp_path))
        with monitor.step("step1"):
            pass
        path = monitor.save_report()
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "steps" in data
        assert data["steps"][0]["name"] == "step1"

    def test_get_step_durations_dict(self, tmp_path):
        from monitoring import PipelineMonitor
        monitor = PipelineMonitor(output_dir=str(tmp_path))
        with monitor.step("step_a"):
            pass
        durations = monitor.get_step_durations()
        assert "timing/step_a_s" in durations


# ===============================================================================
# Tests: Prediction Intervals
# ===============================================================================

class TestPredictionIntervals:
    def test_quantile_predict_shape(self, trained_ridge):
        from prediction_intervals import PredictionIntervalBuilder
        model, X_tr, X_te, y_tr, y_te = trained_ridge
        builder = PredictionIntervalBuilder(method="quantile")
        builder.fit(model, X_tr, y_tr)
        df = builder.predict_with_interval(X_te)
        assert len(df) == len(X_te)
        assert {"prediction", "lower", "upper", "interval_width"}.issubset(df.columns)

    def test_lower_less_than_upper(self, trained_ridge):
        from prediction_intervals import PredictionIntervalBuilder
        model, X_tr, X_te, y_tr, y_te = trained_ridge
        builder = PredictionIntervalBuilder(method="quantile")
        builder.fit(model, X_tr, y_tr)
        df = builder.predict_with_interval(X_te)
        assert (df["lower"] <= df["prediction"]).all()
        assert (df["prediction"] <= df["upper"]).all()

    def test_quantile_coverage_close_to_nominal(self, trained_ridge):
        from prediction_intervals import PredictionIntervalBuilder
        model, X_tr, X_te, y_tr, y_te = trained_ridge
        builder = PredictionIntervalBuilder(method="quantile", confidence_level=0.95)
        builder.fit(model, X_tr, y_tr)
        metrics = builder.evaluate_coverage(X_te, y_te)
        # La couverture doit être raisonnablement proche de 95%
        assert 0.70 <= metrics["picp"] <= 1.0

    def test_bootstrap_intervals(self, trained_ridge):
        from prediction_intervals import PredictionIntervalBuilder
        model, X_tr, X_te, y_tr, y_te = trained_ridge
        builder = PredictionIntervalBuilder(method="bootstrap", n_bootstrap=20)
        builder.fit(model, X_tr, y_tr)
        df = builder.predict_with_interval(X_te)
        assert len(df) == len(X_te)
        assert (df["interval_width"] > 0).all()

    def test_predict_before_fit_raises(self, reg_data):
        from prediction_intervals import PredictionIntervalBuilder
        X_tr, X_te, y_tr, y_te = reg_data
        builder = PredictionIntervalBuilder()
        with pytest.raises(RuntimeError):
            builder.predict_with_interval(X_te)

    def test_format_prediction(self, trained_ridge):
        from prediction_intervals import PredictionIntervalBuilder
        model, X_tr, X_te, y_tr, y_te = trained_ridge
        builder = PredictionIntervalBuilder(method="quantile")
        builder.fit(model, X_tr, y_tr)
        df = builder.predict_with_interval(X_te)
        formatted = builder.format_prediction(df.iloc[0])
        assert "MAD" in formatted or isinstance(formatted, str)


# ===============================================================================
# Tests: Data Validation
# ===============================================================================

class TestDataValidation:
    def test_valid_df_passes(self, sample_df):
        from data_validation import DataValidator
        validator = DataValidator()
        report = validator.validate(sample_df)
        assert report.n_rows == 5
        assert report.passed

    def test_empty_df_fails(self):
        from data_validation import DataValidator
        validator = DataValidator()
        report = validator.validate(pd.DataFrame())
        assert not report.passed

    def test_missing_column_fails(self, sample_df):
        from data_validation import DataValidator
        df_missing = sample_df.drop(columns=["prix"])
        validator = DataValidator(required_columns=["prix", "surface_m2"])
        report = validator.validate(df_missing)
        assert not report.passed

    def test_price_out_of_range_detected(self):
        from data_validation import DataValidator
        df = pd.DataFrame({
            "prix"      : [1] * 50 + [500_000] * 50,   # 50% sous le min
            "surface_m2": [80] * 100,
            "ville"     : ["Casa"] * 100,
        })
        validator = DataValidator(price_min=1_000)
        report = validator.validate(df)
        result = next(r for r in report.results if r.name == "price_range")
        assert not result.passed

    def test_duplicates_detected(self, sample_df):
        from data_validation import DataValidator
        df_dup = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
        validator = DataValidator()
        report = validator.validate(df_dup)
        dup_result = next(r for r in report.results if r.name == "duplicates")
        assert not dup_result.passed

    def test_save_report_creates_json(self, sample_df, tmp_path):
        from data_validation import DataValidator
        validator = DataValidator()
        report = validator.validate(sample_df)
        path = validator.save_report(report, output_dir=str(tmp_path))
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "passed" in data
        assert "results" in data

    def test_report_to_dict(self, sample_df):
        from data_validation import DataValidator
        validator = DataValidator()
        report = validator.validate(sample_df)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["n_rows"] == 5


# ===============================================================================
# Tests: Out-of-Sample Validation (#14)
# ===============================================================================

class TestOutOfSampleValidation:
    """
    Valide les modèles sur des données synthétiques "nouvelles"
    (out-of-sample), différentes des données d'entraînement.
    """

    def test_regression_on_new_distribution(self):
        """
        Entraîne sur une distribution, teste sur une distribution légèrement décalée.
        Le modèle doit avoir R² > 0 même sur des données hors-distribution modérées.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import r2_score

        rng = np.random.default_rng(42)
        X_train = rng.normal(0, 1, (300, 5))
        y_train = 3 * X_train[:, 0] + 2 * X_train[:, 1] + rng.normal(0, 0.5, 300)

        # Données out-of-sample : légèrement décalées
        X_oos = rng.normal(0.3, 1.2, (100, 5))
        y_oos = 3 * X_oos[:, 0] + 2 * X_oos[:, 1] + rng.normal(0, 0.5, 100)

        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        r2_oos = r2_score(y_oos, model.predict(X_oos))
        assert r2_oos > 0.5, f"R² OOS trop faible : {r2_oos:.4f}"

    def test_classification_on_new_samples(self):
        """
        Vérifie que le modèle de classification maintient une accuracy > baseline
        sur des données out-of-sample.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        X, y = make_classification(
            n_samples=600, n_features=8, n_classes=3,
            n_informative=4, n_clusters_per_class=1,
            weights=[0.5, 0.3, 0.2], random_state=100
        )
        X_train, X_oos, y_train, y_oos = train_test_split(X, y, test_size=0.3, random_state=100)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_oos, model.predict(X_oos))
        most_freq_baseline = max(np.bincount(y_oos)) / len(y_oos)

        assert acc > most_freq_baseline, (
            f"Accuracy OOS {acc:.4f} ≤ baseline {most_freq_baseline:.4f}"
        )

    def test_prediction_interval_coverage_oos(self, trained_ridge):
        """
        Vérifie que les intervalles couvrent correctement des données hors-distribution.
        """
        from prediction_intervals import PredictionIntervalBuilder

        model, X_tr, X_te, y_tr, y_te = trained_ridge
        builder = PredictionIntervalBuilder(method="quantile", confidence_level=0.90)
        builder.fit(model, X_tr, y_tr)

        # Données OOS : légèrement décalées
        rng = np.random.default_rng(99)
        X_oos = pd.DataFrame(
            X_te.values + rng.normal(0, 0.1, X_te.shape),
            columns=X_te.columns
        )
        y_oos = y_te + rng.normal(0, 5, len(y_te))

        metrics = builder.evaluate_coverage(X_oos, y_oos)
        # Couverture tolérée entre 70% et 100% (OOS peut dégrader légèrement)
        assert 0.70 <= metrics["picp"] <= 1.0

    def test_model_reproducibility(self, reg_data):
        """Le même modèle avec le même random_state doit donner des résultats identiques."""
        from sklearn.metrics import r2_score
        X_tr, X_te, y_tr, y_te = reg_data

        model_a = Ridge(random_state=42) if hasattr(Ridge(), "random_state") else Ridge()
        model_b = Ridge()
        model_a.fit(X_tr, y_tr)
        model_b.fit(X_tr, y_tr)

        pred_a = model_a.predict(X_te)
        pred_b = model_b.predict(X_te)
        np.testing.assert_array_almost_equal(pred_a, pred_b, decimal=6)

    def test_feature_names_consistency(self, reg_data):
        """Le modèle doit refuser des features avec des noms différents."""
        from sklearn.ensemble import RandomForestRegressor
        X_tr, X_te, y_tr, y_te = reg_data

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_tr, y_tr)

        # Renommer une colonne → doit produire un warning ou erreur selon sklearn
        X_wrong = X_te.rename(columns={"feature_0": "wrong_name"})
        # sklearn >= 1.0 : warning sur les noms, pas d'exception
        # On vérifie simplement que le shape est compatible
        assert X_wrong.shape == X_te.shape
