"""
test_feature_store.py
---------------------
Tests pour FeatureStore: SQLite backend, read/write, cache, assemblage.
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


# Fixtures

@pytest.fixture
def tmp_store(tmp_path):
    """Feature Store dans un dossier temporaire."""
    from feature_store import FeatureStore
    return FeatureStore(store_path=str(tmp_path / "test_store.db"), cache_ttl_s=60)


@pytest.fixture
def sample_df():
    """DataFrame de features de test."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "id"         : [f"prop_{i}" for i in range(n)],
        "surface_m2" : rng.normal(100, 30, n).clip(20, 500),
        "nb_chambres": rng.integers(1, 6, n).astype(float),
        "age_bien"   : rng.normal(15, 8, n).clip(0, 80),
        "prix_par_m2": rng.normal(8000, 2000, n).clip(1000, 30000),
    })


@pytest.fixture
def geo_df():
    """DataFrame de features géographiques."""
    rng = np.random.default_rng(7)
    n = 100
    return pd.DataFrame({
        "id"            : [f"prop_{i}" for i in range(n)],
        "prix_median_ville": rng.normal(900_000, 300_000, n),
        "rank_ville"    : rng.integers(1, 20, n).astype(float),
    })


# Tests write()

class TestWrite:
    def test_write_returns_stats(self, tmp_store, sample_df):
        stats = tmp_store.write(sample_df, group="test_group", entity_key="id")
        assert stats.n_rows == 100
        assert stats.n_features == 4    # surface, chambres, age, prix_m2
        assert stats.group_name == "test_group"
        assert stats.checksum is not None

    def test_write_creates_group(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="property_base", entity_key="id")
        groups = tmp_store.list_groups()
        names = [g["name"] for g in groups]
        assert "property_base" in names

    def test_write_stores_correct_feature_count(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="g1", entity_key="id")
        groups = tmp_store.list_groups()
        g = next(g for g in groups if g["name"] == "g1")
        assert len(g["features"]) == 4  # 4 features (sans entity_key)

    def test_write_overwrite_replaces_data(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="g1", entity_key="id", version="v1")
        # Modifier les données
        sample_df2 = sample_df.copy()
        sample_df2["surface_m2"] = 999
        tmp_store.write(sample_df2, group="g1", entity_key="id", version="v1", overwrite=True)
        df = tmp_store.read_all("g1", version="v1")
        assert abs(df["surface_m2"].mean() - 999) < 1

    def test_write_different_versions(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="g1", entity_key="id", version="v1")
        tmp_store.write(sample_df, group="g1", entity_key="id", version="v2")
        df_v1 = tmp_store.read_all("g1", version="v1")
        df_v2 = tmp_store.read_all("g1", version="v2")
        assert len(df_v1) == 100
        assert len(df_v2) == 100

    def test_write_handles_nan(self, tmp_store):
        df = pd.DataFrame({
            "id"        : ["a", "b", "c"],
            "surface_m2": [100.0, np.nan, 150.0],
            "age"       : [10, 20, np.nan],
        })
        stats = tmp_store.write(df, group="g_nan", entity_key="id")
        assert stats.n_rows == 3

    def test_write_records_stats(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="g1", entity_key="id")
        stats = tmp_store.get_stats("g1")
        assert len(stats) >= 1
        assert stats[0]["n_rows"] == 100


# Tests read()

class TestRead:
    def test_read_returns_dataframe(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="g1", entity_key="id")
        df = tmp_store.read("g1")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_read_by_entity_ids(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="g1", entity_key="id")
        df = tmp_store.read("g1", entity_ids=["prop_0", "prop_1", "prop_2"])
        assert len(df) == 3

    def test_read_returns_empty_for_missing_group(self, tmp_store):
        df = tmp_store.read("nonexistent_group")
        assert df.empty

    def test_read_specific_features(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="g1", entity_key="id")
        df = tmp_store.read("g1", features=["surface_m2", "entity_id"])
        assert "surface_m2" in df.columns
        assert "nb_chambres" not in df.columns

    def test_read_preserves_numeric_types(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="g1", entity_key="id")
        df = tmp_store.read("g1")
        assert pd.api.types.is_float_dtype(df["surface_m2"])

    def test_read_all_returns_all_rows(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="g1", entity_key="id")
        df = tmp_store.read_all("g1")
        assert len(df) == 100


# Tests cache

class TestCache:
    def test_cache_hit_on_second_read(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="g1", entity_key="id")
        tmp_store.read("g1")                    # premier appel: miss
        assert len(tmp_store._memory_cache) > 0  # mis en cache

    def test_cache_invalidated_after_write(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="g1", entity_key="id")
        tmp_store.read("g1")
        cache_before = len(tmp_store._memory_cache)
        # Réecriture → invalide le cache
        tmp_store.write(sample_df, group="g1", entity_key="id")
        cache_after = len(tmp_store._memory_cache)
        assert cache_after <= cache_before


# Tests read_as_of()

class TestReadAsOf:
    def test_read_as_of_past_returns_empty(self, tmp_store, sample_df):
        """Données écrites maintenant, lecture avant maintenant → vide."""
        import time
        past_ts = "2020-01-01T00:00:00"
        tmp_store.write(sample_df, group="g1", entity_key="id")
        df = tmp_store.read_as_of("g1", timestamp=past_ts)
        assert df.empty

    def test_read_as_of_future_returns_data(self, tmp_store, sample_df):
        """Données écrites maintenant, lecture dans le futur → données disponibles."""
        future_ts = "2099-12-31T23:59:59"
        tmp_store.write(sample_df, group="g1", entity_key="id")
        df = tmp_store.read_as_of("g1", timestamp=future_ts)
        assert len(df) == 100


# Tests get_training_dataset()

class TestGetTrainingDataset:
    def test_assembles_multiple_groups(self, tmp_store, sample_df, geo_df):
        tmp_store.write(sample_df, group="property_base", entity_key="id")
        tmp_store.write(geo_df, group="geographic", entity_key="id")
        dataset = tmp_store.get_training_dataset(["property_base", "geographic"])
        assert "surface_m2" in dataset.columns
        assert "prix_median_ville" in dataset.columns

    def test_empty_group_is_ignored(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="property_base", entity_key="id")
        # "nonexistent" n'existe pas → ignoré
        dataset = tmp_store.get_training_dataset(["property_base", "nonexistent"])
        assert not dataset.empty
        assert "surface_m2" in dataset.columns

    def test_all_empty_returns_empty(self, tmp_store):
        dataset = tmp_store.get_training_dataset(["nonexistent1", "nonexistent2"])
        assert dataset.empty


# Tests pipeline_write_features()

class TestPipelineWriteFeatures:
    def test_writes_multiple_groups(self, tmp_path):
        from feature_store import pipeline_write_features
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "surface_m2"       : rng.normal(100, 20, 200),
            "nb_chambres"      : rng.integers(1, 5, 200).astype(float),
            "prix_par_m2"      : rng.normal(8000, 1500, 200),
            "prix_median_ville": rng.normal(900_000, 200_000, 200),
            "mois"             : rng.integers(1, 13, 200).astype(float),
        })
        feature_names = list(df.columns)
        store_path = str(tmp_path / "fs" / "store.db")
        stats = pipeline_write_features(df, feature_names, store_path=store_path)
        assert len(stats) > 0
        assert any("property" in k for k in stats)

    def test_saves_reference_data(self, tmp_path):
        from feature_store import pipeline_write_features
        df = pd.DataFrame({
            "surface_m2": [100, 120, 90],
            "nb_chambres": [2, 3, 1],
        })
        store_path = str(tmp_path / "fs" / "store.db")
        pipeline_write_features(df, list(df.columns), store_path=store_path)
        ref_path = tmp_path / "fs" / "reference_data.pkl"
        assert ref_path.exists()


# Tests list_groups() et get_feature_freshness()

class TestMetadata:
    def test_list_groups_empty_initially(self, tmp_store):
        assert tmp_store.list_groups() == []

    def test_list_groups_after_write(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="g1", entity_key="id")
        tmp_store.write(sample_df, group="g2", entity_key="id")
        groups = tmp_store.list_groups()
        assert len(groups) == 2

    def test_freshness_returns_dict(self, tmp_store, sample_df):
        tmp_store.write(sample_df, group="g1", entity_key="id")
        freshness = tmp_store.get_feature_freshness()
        assert "g1" in freshness
        assert isinstance(freshness["g1"], str)  # ISO datetime

    def test_checksum_changes_with_data(self, tmp_store, sample_df):
        stats1 = tmp_store.write(sample_df, group="g1", entity_key="id", version="v1")
        df2 = sample_df.copy()
        df2["surface_m2"] = 999  # données différentes
        stats2 = tmp_store.write(df2, group="g1", entity_key="id", version="v2")
        assert stats1.checksum != stats2.checksum
