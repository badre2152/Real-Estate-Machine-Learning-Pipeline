"""
tests/test_extract.py
---------------------
Tests unitaires pour extract.py.
Utilise des mocks pour ne pas dépendre de PostgreSQL.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch


@pytest.fixture
def minimal_df():
    return pd.DataFrame({
        "prix"      : [300_000, 800_000, 1_200_000],
        "surface_m2": [50, 80, 120],
        "ville"     : ["Casablanca", "Rabat", "Marrakech"],
    })


@pytest.fixture
def df_with_nulls(minimal_df):
    df = minimal_df.copy()
    df.loc[0, "ville"] = None
    return df


# get_db_engine

class TestGetDbEngine:

    def test_returns_engine(self):
        from src.extract import get_db_engine
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        with patch("src.extract.create_engine") as mock_create, \
             patch("src.extract.text"):
            mock_engine = MagicMock()
            mock_engine.connect.return_value = mock_conn
            mock_create.return_value = mock_engine
            engine = get_db_engine()
            assert engine is not None

    def test_uses_env_vars(self, monkeypatch):
        monkeypatch.setenv("DB_HOST", "myhost")
        monkeypatch.setenv("DB_NAME", "mydb")
        monkeypatch.setenv("DB_USER", "admin")
        from src.extract import get_db_engine
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        with patch("src.extract.create_engine") as mock_create, \
             patch("src.extract.text"):
            mock_engine = MagicMock()
            mock_engine.connect.return_value = mock_conn
            mock_create.return_value = mock_engine
            get_db_engine()
            url = mock_create.call_args[0][0]
            assert "myhost" in url
            assert "mydb" in url

    def test_retry_on_failure(self):
        """get_db_engine doit réessayer max_retries fois avant de lever."""
        from src.extract import get_db_engine
        from sqlalchemy.exc import OperationalError
        with patch("src.extract.create_engine") as mock_create:
            mock_create.side_effect = OperationalError("conn", {}, Exception())
            with pytest.raises(RuntimeError, match="tentatives"):
                get_db_engine(max_retries=2, retry_delay=0)
            assert mock_create.call_count == 2


# validate_schema

class TestValidateSchema:

    def test_passes_with_required_cols(self, minimal_df):
        from src.extract import validate_schema
        validate_schema(minimal_df)  # doit ne pas lever

    def test_raises_on_missing_col(self):
        from src.extract import validate_schema
        df = pd.DataFrame({"surface_m2": [50]})
        with pytest.raises(ValueError, match="manquantes"):
            validate_schema(df)

    def test_raises_on_empty_df(self):
        from src.extract import validate_schema
        with pytest.raises(ValueError):
            validate_schema(pd.DataFrame())

    def test_all_required_cols_checked(self):
        from src.extract import validate_schema, REQUIRED_COLUMNS
        # Un df avec toutes les colonnes sauf la dernière
        cols = REQUIRED_COLUMNS[:-1]
        df = pd.DataFrame({c: [1] for c in cols})
        with pytest.raises(ValueError):
            validate_schema(df)


# extract_obt

class TestExtractObt:

    def test_returns_dataframe(self, minimal_df):
        from src.extract import extract_obt
        with patch("src.extract.pd.read_sql", return_value=minimal_df), \
             patch("src.extract.get_db_engine", return_value=MagicMock()):
            df = extract_obt()
            assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, minimal_df):
        from src.extract import extract_obt
        with patch("src.extract.pd.read_sql", return_value=minimal_df), \
             patch("src.extract.get_db_engine", return_value=MagicMock()):
            df = extract_obt()
            assert len(df) == 3

    def test_required_columns_present(self, minimal_df):
        from src.extract import extract_obt, REQUIRED_COLUMNS
        with patch("src.extract.pd.read_sql", return_value=minimal_df), \
             patch("src.extract.get_db_engine", return_value=MagicMock()):
            df = extract_obt()
            for col in REQUIRED_COLUMNS:
                assert col in df.columns

    def test_not_empty(self, minimal_df):
        from src.extract import extract_obt
        with patch("src.extract.pd.read_sql", return_value=minimal_df), \
             patch("src.extract.get_db_engine", return_value=MagicMock()):
            df = extract_obt()
            assert not df.empty

    def test_reports_missing_values(self, df_with_nulls, caplog):
        import logging
        from src.extract import extract_obt
        with patch("src.extract.pd.read_sql", return_value=df_with_nulls), \
             patch("src.extract.get_db_engine", return_value=MagicMock()), \
             caplog.at_level(logging.INFO, logger="src.extract"):
            extract_obt()
            # Le log doit mentionner les valeurs manquantes
            assert any("manquante" in r.message.lower() or "%" in r.message
                       for r in caplog.records)

    def test_obt_not_yet_transformed(self, minimal_df):
        """
        L'OBT est nettoyée mais NON transformée pour ML.
        extract_obt ne doit pas créer de colonnes ML (log_prix, etc.)
        """
        from src.extract import extract_obt
        with patch("src.extract.pd.read_sql", return_value=minimal_df), \
             patch("src.extract.get_db_engine", return_value=MagicMock()):
            df = extract_obt()
            ml_cols = ["log_prix", "prix_par_m2", "categorie_prix"]
            for col in ml_cols:
                assert col not in df.columns, (
                    f"'{col}' ne doit pas être créé par extract_obt: "
                    f"les transformations ML se font après extraction."
                )
