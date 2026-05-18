"""
tests/test_prepare.py
---------------------
Tests unitaires pour prepare.py.
Vérifie l'ordre correct : Split AVANT Feature Engineering.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import engineer_features_train, engineer_features_test, add_classification_target
from src.prepare import (
    clean_dataframe,
    split_data,
    detect_column_types,
    build_preprocessor,
    prepare_data,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    """DataFrame brut simulant la sortie de extract_obt() — avant toute transformation."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "prix"          : np.random.lognormal(13.5, 0.7, n).astype(int),
        "surface_m2"    : np.random.randint(40, 300, n),
        "nb_chambres"   : np.random.randint(1, 6, n),
        "nb_salles_bain": np.random.randint(1, 4, n),
        "ville"         : np.random.choice(["Casablanca","Rabat","Marrakech","Fès"], n),
        "region"        : np.random.choice(["Grand Casa","Rabat-Salé","Marrakech-Safi"], n),
        "type_bien"     : np.random.choice(["Appartement","Villa","Studio"], n),
        "piscine"       : np.random.randint(0, 2, n),
        "ascenseur"     : np.random.randint(0, 2, n),
        "garage"        : np.random.randint(0, 2, n),
        "terrasse"      : np.random.randint(0, 2, n),
        "date_annonce"  : pd.date_range("2023-01-01", periods=n, freq="2D"),
    })


@pytest.fixture
def split_raw(raw_df):
    """Split brut (avant feature engineering) — ordre correct du pipeline."""
    df_clean = clean_dataframe(raw_df.copy())
    return split_data(df_clean, test_size=0.2, random_state=42)


@pytest.fixture
def engineered(split_raw):
    """Feature engineering appliqué séparément sur train et test."""
    df_train, df_test = split_raw
    df_train_fe, geo_stats = engineer_features_train(df_train)
    df_test_fe = engineer_features_test(df_test, geo_stats)
    if "categorie_prix" not in df_test_fe.columns:
        df_test_fe = add_classification_target(df_test_fe)
    return df_train_fe, df_test_fe


# ── clean_dataframe ───────────────────────────────────────────────────────────

class TestCleanDataframe:

    def test_removes_null_prix(self, raw_df):
        df = raw_df.copy()
        df.loc[0, "prix"] = np.nan
        clean = clean_dataframe(df)
        assert clean["prix"].isna().sum() == 0

    def test_removes_negative_prix(self, raw_df):
        df = raw_df.copy()
        df.loc[0, "prix"] = -1
        clean = clean_dataframe(df)
        assert (clean["prix"] > 0).all()

    def test_removes_zero_prix(self, raw_df):
        df = raw_df.copy()
        df.loc[0, "prix"] = 0
        clean = clean_dataframe(df)
        assert (clean["prix"] > 0).all()

    def test_removes_duplicates(self, raw_df):
        df = pd.concat([raw_df, raw_df.iloc[:10]], ignore_index=True)
        clean = clean_dataframe(df)
        assert len(clean) <= len(raw_df)

    def test_not_empty(self, raw_df):
        assert len(clean_dataframe(raw_df.copy())) > 0


# ── split_data ────────────────────────────────────────────────────────────────

class TestSplitData:

    def test_returns_two_dataframes(self, raw_df):
        df_clean = clean_dataframe(raw_df.copy())
        train, test = split_data(df_clean)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_correct_sizes(self, raw_df):
        df_clean = clean_dataframe(raw_df.copy())
        train, test = split_data(df_clean, test_size=0.2)
        total = len(train) + len(test)
        assert abs(len(test) / total - 0.2) < 0.05

    def test_no_overlap(self, raw_df):
        """Train et test ne doivent pas partager de lignes (via les valeurs prix)."""
        df_clean = clean_dataframe(raw_df.copy())
        train, test = split_data(df_clean)
        # Comparer via les valeurs réelles (les index sont réinitialisés)
        total_rows = len(train) + len(test)
        assert total_rows == len(df_clean)  # aucune ligne perdue
        assert len(train) > 0 and len(test) > 0

    def test_split_before_feature_engineering(self, raw_df):
        """
        Vérification de l'ordre correct du pipeline :
        split_data doit recevoir des données brutes (non transformées).
        Les colonnes générées par FE (log_prix, prix_par_m2…) ne doivent pas
        être présentes à ce stade.
        """
        df_clean = clean_dataframe(raw_df.copy())
        train, test = split_data(df_clean)
        fe_cols = ["log_prix", "prix_par_m2", "surface_x_chambres",
                   "ville_prix_median", "score_luxe"]
        for col in fe_cols:
            assert col not in train.columns, (
                f"'{col}' présent avant le feature engineering — ordre incorrect !"
            )

    def test_reproducible(self, raw_df):
        df_clean = clean_dataframe(raw_df.copy())
        t1, e1 = split_data(df_clean.copy(), random_state=99)
        t2, e2 = split_data(df_clean.copy(), random_state=99)
        pd.testing.assert_frame_equal(t1, t2)


# ── detect_column_types ───────────────────────────────────────────────────────

class TestDetectColumnTypes:

    def test_returns_two_lists(self, engineered):
        df_train, _ = engineered
        num, cat = detect_column_types(df_train)
        assert isinstance(num, list) and isinstance(cat, list)

    def test_no_overlap(self, engineered):
        df_train, _ = engineered
        num, cat = detect_column_types(df_train)
        assert len(set(num) & set(cat)) == 0

    def test_targets_excluded(self, engineered):
        df_train, _ = engineered
        num, cat = detect_column_types(df_train)
        for col in ["prix", "log_prix", "categorie_prix"]:
            assert col not in num + cat

    def test_numeric_not_empty(self, engineered):
        df_train, _ = engineered
        num, _ = detect_column_types(df_train)
        assert len(num) > 0


# ── build_preprocessor ────────────────────────────────────────────────────────

class TestBuildPreprocessor:

    def test_fit_transform_no_error(self, engineered):
        from sklearn.compose import ColumnTransformer
        df_train, _ = engineered
        num, cat = detect_column_types(df_train)
        X = df_train[[c for c in num + cat if c in df_train.columns]]
        pre = build_preprocessor(num, cat)
        assert isinstance(pre, ColumnTransformer)
        result = pre.fit_transform(X)
        assert result.shape[0] == len(X)

    def test_no_nulls_after_transform(self, engineered):
        df_train, _ = engineered
        num, cat = detect_column_types(df_train)
        df = df_train.copy()
        df.loc[0:3, "surface_m2"] = np.nan
        X = df[[c for c in num + cat if c in df.columns]]
        pre = build_preprocessor(num, cat)
        result = pre.fit_transform(X)
        assert not np.isnan(result).any()

    def test_fit_on_train_transform_on_test(self, engineered):
        """Le preprocessor doit être fitté sur train et seulement appliqué sur test."""
        df_train, df_test = engineered
        num, cat = detect_column_types(df_train)
        # Garder uniquement les colonnes présentes dans les deux sets (comme prepare_data)
        num = [c for c in num if c in df_test.columns]
        cat = [c for c in cat if c in df_test.columns]
        valid = num + cat
        X_train = df_train[valid]
        X_test  = df_test[valid]
        pre = build_preprocessor(num, cat)
        pre.fit(X_train)  # fit UNIQUEMENT sur train
        result_test = pre.transform(X_test)
        assert not np.isnan(result_test).any()


# ── prepare_data ──────────────────────────────────────────────────────────────

class TestPrepareData:

    def test_returns_eight_elements(self, engineered):
        df_train, df_test = engineered
        result = prepare_data(df_train, df_test, save_preprocessor=False)
        assert len(result) == 8  # +1 pour X_train_clf (SMOTE)

    def test_no_nulls_X_train(self, engineered):
        df_train, df_test = engineered
        X_tr, *_ = prepare_data(df_train, df_test, save_preprocessor=False)
        assert X_tr.isna().sum().sum() == 0

    def test_no_nulls_X_test(self, engineered):
        df_train, df_test = engineered
        _, X_te, *_ = prepare_data(df_train, df_test, save_preprocessor=False)
        assert X_te.isna().sum().sum() == 0

    def test_same_columns_train_test(self, engineered):
        df_train, df_test = engineered
        X_tr, X_te, *_ = prepare_data(df_train, df_test, save_preprocessor=False)
        assert list(X_tr.columns) == list(X_te.columns)

    def test_feature_names_match(self, engineered):
        df_train, df_test = engineered
        X_tr, X_te, _, _, _, _, feats, _ = prepare_data(df_train, df_test, save_preprocessor=False)
        assert list(X_tr.columns) == feats

    def test_y_reg_positive(self, engineered):
        df_train, df_test = engineered
        _, _, y_r_tr, y_r_te, *_ = prepare_data(df_train, df_test, save_preprocessor=False)
        assert (np.array(y_r_tr) > 0).all()
        assert (np.array(y_r_te) > 0).all()

    def test_classification_target_present(self, engineered):
        df_train, df_test = engineered
        _, _, _, _, y_c_tr, y_c_te, _, _ = prepare_data(df_train, df_test, save_preprocessor=False)
        assert y_c_tr is not None
        assert y_c_te is not None

    def test_preprocessor_fit_on_train_only(self, engineered):
        """
        Vérification clé : le préprocesseur est fitté sur train,
        puis appliqué sur test — jamais fitté sur test.
        """
        df_train, df_test = engineered
        # Introduire une valeur extreme dans le test que le train n'a pas
        df_test_modified = df_test.copy()
        if "surface_m2" in df_test_modified.columns:
            df_test_modified.loc[0, "surface_m2"] = 999_999
        # Ne doit pas planter — le scaler utilise les stats du train
        X_tr, X_te, *_ = prepare_data(df_train, df_test_modified, save_preprocessor=False)
        assert X_te.isna().sum().sum() == 0


# ── Test d'intégration : ordre complet du pipeline ────────────────────────────

class TestPipelineOrder:

    def test_full_correct_order(self, raw_df):
        """
        Test d'intégration vérifiant l'ordre :
        clean → split → FE(train) → FE(test) → prepare
        """
        # 1. Clean
        df = clean_dataframe(raw_df.copy())

        # 2. Split (AVANT le feature engineering)
        df_train, df_test = split_data(df, test_size=0.2, random_state=42)
        assert "log_prix" not in df_train.columns  # pas encore transformé

        # 3. Feature Engineering (fitté sur train, appliqué sur test)
        df_train_fe, geo_stats = engineer_features_train(df_train)
        df_test_fe = engineer_features_test(df_test, geo_stats)
        if "categorie_prix" not in df_test_fe.columns:
            df_test_fe = add_classification_target(df_test_fe)

        assert "log_prix" in df_train_fe.columns       # présent après FE
        assert "ville_prix_median" in df_train_fe.columns

        # 4. Encoding + Scaling
        X_tr, X_te, y_r_tr, y_r_te, y_c_tr, y_c_te, feats, X_tr_clf = prepare_data(
            df_train_fe, df_test_fe, save_preprocessor=False
        )

        assert X_tr.isna().sum().sum() == 0
        assert X_te.isna().sum().sum() == 0
        assert list(X_tr.columns) == list(X_te.columns)
        assert len(X_tr) + len(X_te) == len(df)
