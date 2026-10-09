"""Check serving inputs against the saved preprocessing contract."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api import _build_input_df
from prepare import build_preprocessor, detect_column_types


def test_api_input_matches_training_preprocessor():
    train = pd.DataFrame({
        "surface_m2": [40.0, 60.0, 80.0, 100.0],
        "ville": ["Casa", "Rabat", "Fes", "Casa"],
        "quartier": ["Centre", "Nord", "Sud", "Centre"],
        "nb_chambres": [1, 2, 3, 4],
        "nb_salles_bain": [1, 1, 2, 2],
        "etage": [1, 2, 3, 4],
        "surface_x_chambres": [40, 120, 240, 400],
        "surface_par_chambre": [20, 20, 20, 20],
        "ratio_chambres_bains": [0.5, 1.0, 1.0, 4 / 3],
    })
    numeric, categorical = detect_column_types(train)
    preprocessor = build_preprocessor(numeric, categorical)
    trained = preprocessor.fit_transform(train[numeric + categorical])
    model = Ridge().fit(trained, [100000, 160000, 220000, 280000])

    sample = SimpleNamespace(
        surface_m2=70.0, ville="Casa", quartier="Centre",
        type_bien="Appartement", nb_chambres=2, nb_salles_bain=1,
        etage=2, age_bien=None,
    )
    inference = _build_input_df(sample)
    transformed = preprocessor.transform(inference)
    prediction = model.predict(transformed)

    assert transformed.shape[1] == trained.shape[1]
    assert np.isfinite(prediction).all()
