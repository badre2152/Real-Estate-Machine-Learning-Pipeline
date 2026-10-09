"""Verify fold-local preprocessing during model selection."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import regression


def test_cross_validation_receives_unfitted_preprocessing(monkeypatch):
    raw = pd.DataFrame({
        "surface_m2": [float(i) for i in range(10, 30)],
        "ville": ["Casa", "Rabat"] * 10,
        "prix": [float(i * 1000) for i in range(10, 30)],
    })
    prepared = pd.DataFrame({
        "surface_m2": raw["surface_m2"].to_numpy(),
        "ville_Casa": [1, 0] * 10,
        "ville_Rabat": [0, 1] * 10,
    })
    inspected = []

    def verify_cv(model, X, y, **kwargs):
        assert isinstance(model, Pipeline)
        assert isinstance(model.named_steps["estimator"], Ridge)
        assert list(X.columns) == ["surface_m2", "ville"]
        assert not hasattr(model.named_steps["preprocessor"], "transformers_")
        inspected.append(True)
        return np.array([0.5, 0.6])

    monkeypatch.setattr(regression, "get_regression_models", lambda: {"Ridge": Ridge()})
    monkeypatch.setattr(regression, "cross_val_score", verify_cv)
    model, name, log_target = regression.train_regression(
        prepared, raw["prix"], cv_frame=raw
    )

    assert inspected == [True]
    assert name == "Ridge"
    assert log_target is False
    assert hasattr(model, "coef_")
