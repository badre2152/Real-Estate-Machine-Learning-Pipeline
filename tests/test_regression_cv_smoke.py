"""Exercise actual cross-validation with fold-local preprocessing."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import regression
from prepare import build_preprocessor, detect_column_types


def test_regression_model_selection_with_real_cv(monkeypatch):
    raw = pd.DataFrame({
        "surface_m2": np.arange(40, 70, dtype=float),
        "ville": ["Casa", "Rabat", "Fes"] * 10,
        "prix": np.arange(40, 70, dtype=float) * 2000,
    })
    numeric, categorical = detect_column_types(raw)
    preprocessor = build_preprocessor(numeric, categorical)
    prepared = preprocessor.fit_transform(raw[numeric + categorical])
    processed = pd.DataFrame(
        prepared, columns=preprocessor.get_feature_names_out()
    )

    monkeypatch.setattr(regression, "get_regression_models", lambda: {"Ridge": Ridge()})
    monkeypatch.setattr(regression, "_CV", 3)
    model, name, uses_log_target = regression.train_regression(
        processed, raw["prix"], cv_frame=raw
    )

    assert name == "Ridge"
    assert uses_log_target is False
    assert np.isfinite(model.predict(processed)).all()
