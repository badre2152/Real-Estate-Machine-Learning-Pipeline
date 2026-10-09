"""Ensure target-derived predictors never enter the serving matrix."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare import detect_column_types


def test_price_derived_features_are_not_model_inputs():
    frame = pd.DataFrame({
        "prix": [100000, 200000],
        "surface_m2": [50, 70],
        "ville": ["Casa", "Rabat"],
        "prix_par_m2": [2000, 2857],
        "log_prix": [11.5, 12.2],
        "ecart_prix_ville": [5000, 10000],
        "ville_prix_median": [95000, 190000],
        "region_prix_median": [90000, 180000],
        "surface_x_chambres": [100, 210],
    })

    numeric, categorical = detect_column_types(frame)
    selected = set(numeric + categorical)

    assert selected == {"surface_m2", "ville", "surface_x_chambres"}
    assert "prix" not in selected
    assert "prix_par_m2" not in selected
    assert "ville_prix_median" not in selected
