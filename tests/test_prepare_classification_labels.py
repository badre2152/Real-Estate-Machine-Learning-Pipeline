"""Regression tests for classification data validation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare import prepare_data


def test_prepare_data_rejects_missing_classification_labels():
    train = pd.DataFrame({
        "prix": [120000, 150000, 180000, 200000],
        "surface_m2": [50, 60, 70, 80],
        "categorie_prix": [None, None, None, None],
    })
    test = pd.DataFrame({
        "prix": [130000, 170000],
        "surface_m2": [55, 65],
        "categorie_prix": ["Low", "High"],
    })

    with pytest.raises(ValueError, match="Classification labels contain no valid values"):
        prepare_data(train, test, save_preprocessor=False)
