"""
features.py
-----------
Feature Engineering pour le pipeline ML immobilier.

ORDRE CORRECT selon le contexte :
  Extraction OBT → Split → Feature Engineering → Scaling/Encoding → Training

DONC : les features qui calculent des statistiques sur le dataset (ex: prix médian
par ville) DOIVENT être calculées uniquement sur le train set, puis appliquées au
test set: pour éviter la fuite de données (data leakage).

Les fonctions de ce module sont donc appelées APRÈS le split.
Les fonctions stateless (log, ratio, interaction) peuvent s'appliquer librement.

Fonctions stateless (safe sur train+test séparément) :
  - add_log_price, add_price_per_m2, add_surface_rooms_interaction,
    add_luxury_score, add_temporal_features, add_price_zscore

Fonctions stateful (doivent être fittées sur train, appliquées sur test) :
  - add_geographic_features → fit_geographic_stats() + apply_geographic_stats()
"""

from logger_setup import get_logger
import numpy as np
import pandas as pd
from scipy import stats

logger = get_logger(__name__)

TARGET_REGRESSION     = "prix"
TARGET_CLASSIFICATION = "categorie_prix"

def add_log_price(df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
    """
    Log1p transformation du prix: distribution plus gaussienne.

    ATTENTION Data Leakage :
      log_prix = log(prix) utilise la TARGET.
      → Train uniquement. Omis en mode inference.
    """
    if is_inference:
        return df
    if TARGET_REGRESSION in df.columns:
        df["log_prix"] = np.log1p(df[TARGET_REGRESSION])
        logger.info("   ✅ log_prix créé (train uniquement)")
    return df

def add_price_per_m2(df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
    """
    Prix au m²: feature clé en immobilier. +1 évite la division par zéro.

    ATTENTION Data Leakage :
      prix_par_m2 = prix / surface_m2 utilise la TARGET.
      → Calculée sur train : OK (supervisé, target connue).
      → Sur test / inference : target inconnue → cette feature est OMISE.
      Le préprocesseur sklearn gère l'absence via handle_unknown="ignore".

    Args:
        is_inference: True = mode API/prediction (pas de target) → feature omise.
    """
    if is_inference:
        logger.debug("   ℹ️  prix_par_m2 omis en mode inference (target inconnue)")
        return df

    if TARGET_REGRESSION in df.columns and "surface_m2" in df.columns:
        df["prix_par_m2"]     = df[TARGET_REGRESSION] / (df["surface_m2"] + 1)
        df["log_prix_par_m2"] = np.log1p(df["prix_par_m2"])
        logger.info("   ✅ prix_par_m2 + log_prix_par_m2 créés (train uniquement)")
    return df

def add_surface_rooms_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Interactions surface × pièces."""
    if "surface_m2" in df.columns and "nb_chambres" in df.columns:
        df["surface_x_chambres"]  = df["surface_m2"] * df["nb_chambres"]
        df["surface_par_chambre"] = df["surface_m2"] / (df["nb_chambres"] + 1)
        logger.info("   ✅ surface_x_chambres + surface_par_chambre créés")
    if "nb_chambres" in df.columns and "nb_salles_bain" in df.columns:
        df["ratio_chambres_bains"] = df["nb_chambres"] / (df["nb_salles_bain"] + 1)
        logger.info("   ✅ ratio_chambres_bains créé")
    return df

def add_luxury_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score composite d'équipements haut de gamme."""
    luxury_cols = ["piscine", "ascenseur", "garage", "concierge",
                   "terrasse", "jardin", "climatisation"]
    existing = [c for c in luxury_cols if c in df.columns]
    if existing:
        df["score_luxe"] = df[existing].fillna(0).astype(int).sum(axis=1)
        logger.info(f"   ✅ score_luxe créé ({len(existing)} équipements)")
    return df

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables temporelles dérivées depuis la date d'annonce."""
    date_col = next(
        (c for c in ["date_annonce", "created_at", "date_scraping"] if c in df.columns),
        None,
    )
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["mois_annonce"]  = df[date_col].dt.month
        df["trimestre"]     = df[date_col].dt.quarter
        df["est_weekend"]   = (df[date_col].dt.dayofweek >= 5).astype(int)
        df["jours_depuis_annonce"] = (
            df[date_col].max() - df[date_col]
        ).dt.days.clip(lower=0)
        logger.info(f"   ✅ Features temporelles créées depuis '{date_col}'")
    else:
        logger.warning("   ⚠️  Pas de colonne date: features temporelles ignorées")
    return df

def fit_geographic_stats(df_train: pd.DataFrame) -> dict:
    """
    Calcule les statistiques géographiques sur le TRAIN SET uniquement.
    Retourne un dict de stats à appliquer ensuite sur train ET test.

    ANTI-DATA-LEAKAGE : ne jamais appeler sur le dataset complet avant le split.
    """
    stats_dict = {}

    if TARGET_REGRESSION in df_train.columns and "ville" in df_train.columns:
        city_stats = (
            df_train.groupby("ville")[TARGET_REGRESSION]
            .agg(ville_prix_median="median", ville_prix_mean="mean")
        )
        stats_dict["city_stats"] = city_stats
        stats_dict["global_median"] = df_train[TARGET_REGRESSION].median()

        rank_map = city_stats["ville_prix_median"].rank(ascending=False).to_dict()
        stats_dict["city_rank"] = rank_map
        logger.info(f"   ✅ Stats géo fittées sur {len(df_train):,} lignes train")

    if TARGET_REGRESSION in df_train.columns and "region" in df_train.columns:
        region_stats = df_train.groupby("region")[TARGET_REGRESSION].median()
        stats_dict["region_stats"]  = region_stats
        stats_dict["global_median"] = df_train[TARGET_REGRESSION].median()

    return stats_dict

def apply_geographic_stats(df: pd.DataFrame, geo_stats: dict) -> pd.DataFrame:
    """
    Applique les stats géographiques pré-calculées (depuis fit_geographic_stats)
    sur n'importe quel DataFrame (train ou test).
    """
    global_median = geo_stats.get("global_median", 0)

    if "city_stats" in geo_stats and "ville" in df.columns:
        city_stats = geo_stats["city_stats"]
        df = df.join(city_stats, on="ville", how="left")
        df["ville_prix_median"] = df["ville_prix_median"].fillna(global_median)
        df["ville_prix_mean"]   = df["ville_prix_mean"].fillna(global_median)

        rank_map = geo_stats.get("city_rank", {})
        df["ville_rang_prix"] = df["ville"].map(rank_map).fillna(rank_map and max(rank_map.values()) + 1 or 999)
        logger.info("   ✅ Stats géo appliquées")

    if "region_stats" in geo_stats and "region" in df.columns:
        region_stats = geo_stats["region_stats"].rename("region_prix_median")
        df = df.join(region_stats, on="region", how="left")
        df["region_prix_median"] = df["region_prix_median"].fillna(global_median)

    return df

def add_classification_target(df: pd.DataFrame) -> pd.DataFrame:
    if "type_bien" not in df.columns:
        return df.drop(columns=[TARGET_CLASSIFICATION], errors="ignore")
    df[TARGET_CLASSIFICATION] = df["type_bien"].astype("string")
    return df

def apply_stateless_features(df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
    """
    Applique toutes les features stateless.

    Args:
        is_inference: True en mode API/test: exclut les features dérivées de la target
                      (prix_par_m2, log_prix) pour éviter le data leakage.

    Features toujours calculées (safe sur train + test + inference) :
      add_surface_rooms_interaction, add_luxury_score, add_temporal_features

    Features train-only (utilisent la target) :
      add_log_price, add_price_per_m2
    """
    df = add_surface_rooms_interaction(df)
    df = add_luxury_score(df)
    df = add_temporal_features(df)

    df = add_log_price(df, is_inference=is_inference)
    df = add_price_per_m2(df, is_inference=is_inference)

    return df

def engineer_features_train(df_train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Lance le feature engineering COMPLET sur le train set.
    Retourne le DataFrame enrichi + les stats géo pour application sur le test.

    À appeler APRÈS le split.
    """
    logger.info("\n" + "=" * 50)
    logger.info("⚙️  FEATURE ENGINEERING: TRAIN SET")
    logger.info("=" * 50)

    n_before = df_train.shape[1]
    df_train = add_classification_target(df_train)
    df_train = apply_stateless_features(df_train)

    geo_stats = fit_geographic_stats(df_train)
    df_train  = apply_geographic_stats(df_train, geo_stats)

    logger.info(f"\n✅ Train FE terminé : {n_before} → {df_train.shape[1]} colonnes")
    return df_train, geo_stats

def engineer_features_test(df_test: pd.DataFrame, geo_stats: dict) -> pd.DataFrame:
    """
    Lance le feature engineering sur le test set en utilisant les stats du train.
    À appeler APRÈS engineer_features_train().

    Passe is_inference=True pour exclure les features dérivées de la target
    (prix_par_m2, log_prix): ces features ne sont pas calculables au moment
    de la prédiction car la target est inconnue → anti data leakage.
    """
    logger.info("\n⚙️  FEATURE ENGINEERING: TEST SET")
    df_test = apply_stateless_features(df_test, is_inference=True)
    df_test = apply_geographic_stats(df_test, geo_stats)
    logger.info(f"✅ Test FE terminé : {df_test.shape[1]} colonnes")
    return df_test
