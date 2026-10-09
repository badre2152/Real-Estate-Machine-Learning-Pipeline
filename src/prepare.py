"""
prepare.py
----------
Préparation des données pour le ML (post-extraction OBT).

ORDRE CORRECT selon le contexte du projet :
  Extraction OBT → Split → Feature Engineering → Scaling/Encoding → Training

Le split est donc la PREMIÈRE transformation après l'extraction.
Le feature engineering est fait APRÈS le split pour éviter la fuite de données.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from logger_setup import get_logger

try:
    from config_loader import cfg as _cfg
    _RS = int(_cfg.pipeline.random_state)
    _TS = float(_cfg.pipeline.test_size)
except Exception:
    _RS, _TS = 42, 0.2

logger = get_logger(__name__)

TARGET_REGRESSION     = "prix"
TARGET_CLASSIFICATION = "categorie_prix"

EXCLUDE_FROM_FEATURES = {
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    "log_prix",
    "prix_par_m2",
    "log_prix_par_m2",
    "ecart_prix_ville",
    "ville_prix_mean",
    "ville_prix_median",
    "ville_rang_prix",
    "region_prix_median",
    "id", "url", "titre", "description",
    "date_annonce", "created_at", "date_scraping",  # brutes → remplacées par les dérivées
    "type_bien",          # source de la cible classification
}

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage minimal post-extraction :
    - Supprime les doublons
    - Supprime les lignes sans prix (target obligatoire)
    - Supprime les prix <= 0 (aberrants)
    La table OBT est déjà nettoyée, ce nettoyage est une sécurité.
    """
    logger.info("🧹 Nettoyage de sécurité ...")
    n0 = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_REGRESSION])
    df = df[df[TARGET_REGRESSION] > 0]
    logger.info(f"   Lignes supprimées : {n0 - len(df):,} | Restantes : {len(df):,}")
    return df

def split_data(
    df: pd.DataFrame,
    test_size: float = None,
    random_state: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    PREMIÈRE étape après extraction : split train/test sur le DataFrame brut.
    Toute transformation ultérieure doit être fittée sur train uniquement.

    Returns:
        df_train, df_test: DataFrames bruts (non transformés)
    """
    df_train, df_test = train_test_split(df, test_size=(test_size or _TS), random_state=(random_state or _RS))
    logger.info(f"   ✅ Split : {len(df_train):,} train | {len(df_test):,} test")
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

def detect_column_types(df: pd.DataFrame) -> tuple[list, list]:
    """
    Détecte automatiquement les colonnes numériques et catégorielles,
    en excluant les colonnes non-features.
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]

    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_cols = [
        c for c in feature_cols
        if pd.api.types.is_string_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
        or df[c].dtype == object
    ]

    logger.info(f"   Numériques    : {len(numeric_cols)} | Catégorielles : {len(categorical_cols)}")
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Construit le préprocesseur sklearn :
    - Numériques  : Imputation médiane + StandardScaler
    - Catégorielles : Imputation 'inconnu' + OneHotEncoder

    Note : OneHotEncoder est préféré à LabelEncoder car les catégories
    (ville, type_bien, region) n'ont pas d'ordre naturel.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="inconnu")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")

def apply_smote(X_train: pd.DataFrame, y_train, random_state: int = 42):
    """
    Applique SMOTE sur le train set de classification uniquement.
    Ne modifie PAS X utilisé pour la régression.
    """
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        logger.info(f"   ✅ SMOTE : {len(X_train):,} → {len(X_res):,} échantillons")
        return X_res, y_res
    except ImportError:
        logger.warning("   ⚠️  imbalanced-learn non installé: SMOTE ignoré")
        return X_train, y_train

def prepare_data(
    df_train_fe: pd.DataFrame,
    df_test_fe: pd.DataFrame,
    use_smote: bool = False,
    random_state: int = None,
    save_preprocessor: bool = True,
) -> tuple:
    """
    Prépare les données APRÈS feature engineering.

    Args:
        df_train_fe:       Train set après feature engineering.
        df_test_fe:        Test set après feature engineering (stats du train).
        use_smote:         Appliquer SMOTE si déséquilibre de classes.
        random_state:      Graine aléatoire.
        save_preprocessor: Sauvegarder le préprocesseur pour l'inférence.

    Returns:
        X_train, X_test,
        y_reg_train, y_reg_test,
        y_clf_train, y_clf_test,
        feature_names
    """
    logger.info("\n" + "=" * 50)
    logger.info("🔧 PRÉPARATION: Encoding + Scaling")
    logger.info("=" * 50)

    y_reg_train = df_train_fe[TARGET_REGRESSION].reset_index(drop=True)
    y_reg_test  = df_test_fe[TARGET_REGRESSION].reset_index(drop=True)

    y_clf_train = df_train_fe.get(TARGET_CLASSIFICATION)
    y_clf_test  = df_test_fe.get(TARGET_CLASSIFICATION)
    if y_clf_train is not None:
        y_clf_train = y_clf_train.reset_index(drop=True)
        y_clf_test  = y_clf_test.reset_index(drop=True)

    numeric_cols, categorical_cols = detect_column_types(df_train_fe)

    numeric_cols     = [c for c in numeric_cols     if c in df_test_fe.columns]
    categorical_cols = [c for c in categorical_cols if c in df_test_fe.columns]

    valid_cols = numeric_cols + categorical_cols
    X_train_raw = df_train_fe[valid_cols].reset_index(drop=True)
    X_test_raw  = df_test_fe[valid_cols].reset_index(drop=True)

    preprocessor  = build_preprocessor(numeric_cols, categorical_cols)
    X_train_arr   = preprocessor.fit_transform(X_train_raw)
    X_test_arr    = preprocessor.transform(X_test_raw)

    ohe_names = []
    if categorical_cols and "cat" in preprocessor.named_transformers_:
        ohe = preprocessor.named_transformers_["cat"]["encoder"]
        ohe_names = list(ohe.get_feature_names_out(categorical_cols))
    feature_names = numeric_cols + ohe_names

    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test  = pd.DataFrame(X_test_arr,  columns=feature_names)

    X_train_clf = X_train.copy()
    if y_clf_train is not None:
        counts = y_clf_train.value_counts()
        ratio  = counts.min() / counts.max()
        logger.info(f"   Distribution classes : {counts.to_dict()}")
        if ratio < 0.5:
            logger.warning(f"   ⚠️  Déséquilibre (ratio={ratio:.2f})")
            if use_smote:
                X_train_clf, y_clf_train = apply_smote(X_train_clf, y_clf_train, random_state or _RS)

    if save_preprocessor:
        from config_loader import cfg
        model_dir = os.getenv("MODELS_DIR") or cfg.paths.models_dir
        os.makedirs(model_dir, exist_ok=True)
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)
        logger.info("   Préprocesseur sauvegardé dans %s", preprocessor_path)

    logger.info(f"\n✅ Train : {len(X_train):,} | Test : {len(X_test):,} | Features : {len(feature_names)}")
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, feature_names, X_train_clf