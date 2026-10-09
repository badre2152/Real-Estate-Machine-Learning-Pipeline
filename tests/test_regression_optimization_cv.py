"""Check preprocessing and estimator compatibility in hyperparameter search."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import regression


def test_optimization_uses_raw_features_inside_cv(monkeypatch):
    raw = pd.DataFrame({
        "surface_m2": list(range(30, 50)),
        "ville": ["Casa", "Rabat"] * 10,
        "prix": np.arange(30, 50) * 1000,
    })
    prepared = pd.DataFrame({
        "surface_m2": np.arange(30, 50),
        "ville_Casa": [1, 0] * 10,
        "ville_Rabat": [0, 1] * 10,
    })

    class FakeSearch:
        def __init__(self, estimator, param_distributions, **kwargs):
            assert isinstance(estimator, Pipeline)
            assert "estimator__n_estimators" in param_distributions
            self.estimator = estimator
            self.best_params_ = {"estimator__n_estimators": 10}
            self.best_score_ = 0.75

        def fit(self, X, y):
            assert list(X.columns) == ["surface_m2", "ville"]
            assert not hasattr(self.estimator.named_steps["preprocessor"], "transformers_")
            return self

    monkeypatch.setattr(regression, "RandomizedSearchCV", FakeSearch)
    model = regression.optimize_model(
        RandomForestRegressor(n_estimators=5, random_state=42),
        prepared,
        raw["prix"],
        cv_frame=raw,
    )

    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 10
    assert hasattr(model, "estimators_")
