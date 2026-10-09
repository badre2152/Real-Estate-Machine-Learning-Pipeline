"""Verify saved regression artifacts produce inference-compatible predictions."""

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api import _build_input_df
from prepare import build_preprocessor, detect_column_types
from regression import PriceScaleRegressor


def test_serialized_model_and_preprocessor_support_serving(tmp_path):
    frame = pd.DataFrame({
        "surface_m2": [40.0, 50.0, 65.0, 75.0, 90.0],
        "ville": ["Casa", "Rabat", "Casa", "Fes", "Rabat"],
        "nb_chambres": [1, 1, 2, 2, 3],
    })
    numeric, categorical = detect_column_types(frame)
    preprocessor = build_preprocessor(numeric, categorical)
    features = preprocessor.fit_transform(frame[numeric + categorical])
    estimator = Ridge().fit(features, np.log1p([100000, 115000, 160000, 190000, 220000]))
    wrapper = PriceScaleRegressor(estimator, log_target=True)
    wrapper.estimator_ = estimator

    model_path = tmp_path / "regression_model.pkl"
    prep_path = tmp_path / "preprocessor.pkl"
    model_path.write_bytes(pickle.dumps(wrapper))
    prep_path.write_bytes(pickle.dumps(preprocessor))

    saved_model = pickle.loads(model_path.read_bytes())
    saved_preprocessor = pickle.loads(prep_path.read_bytes())
    sample = SimpleNamespace(
        surface_m2=62.0, ville="Casa", quartier="Centre",
        type_bien="Appartement", nb_chambres=2, nb_salles_bain=1,
        etage=2, age_bien=None,
    )
    features = saved_preprocessor.transform(_build_input_df(sample))
    prediction = saved_model.predict(features)

    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()
    assert (prediction > 0).all()
