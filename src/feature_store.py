"""
feature_store.py
----------------
Feature Store للـ ML Pipeline: Avito Real Estate v3.

واش هو الـ Feature Store؟
  بدل ما كل pipeline يعيد حساب نفس الـ features من الصفر،
  الـ Feature Store يحسبهم مرة واحدة، يخزّنهم، وكل مرة
  pipeline أو API يحتاجهم يجيبهم مباشرة: سريع ومتّسق.

المشكلة اللي يحلّها:
  FAIL قبل : pipeline → OBT → compute features → train
            API     → OBT → compute features → predict
            → نفس الـ features محسوبة مرتين بطريقتين مختلفتين!

  PASS دابا : pipeline → OBT → FeatureStore.write() → train
            API     → FeatureStore.read() → predict
            → نفس الـ features دائماً، مرة واحدة

Architecture :
  ┌=====================================================┐
  │  FeatureStore                                        │
  │                                                      │
  │  Backend: SQLite (local) ou PostgreSQL (production) │
  │                                                      │
  │  Tables:                                             │
  │    feature_groups     → تعريف الـ feature groups    │
  │    feature_definitions → تعريف كل feature           │
  │    feature_values     → القيم المحسوبة              │
  │    feature_stats      → إحصائيات للـ monitoring     │
  └=====================================================│

Feature Groups (منظّمة حسب النوع):
  - property_base      : features الأساسية (surface, nb_chambres...)
  - property_derived   : features محسوبة (prix_par_m2, score_luxe...)
  - geographic         : features الجغرافية (median_city, rank_ville...)
  - temporal           : features الزمنية (mois, annee, saison...)

Usage:
    from feature_store import FeatureStore

    fs = FeatureStore()

    fs.write(df=X_train_with_metadata, group="property_base", version="v1")

    features = fs.read(entity_ids=["id1", "id2"], group="property_base")

    features = fs.read_as_of(timestamp="2024-01-01", group="geographic")
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from logger_setup import get_logger

logger = get_logger(__name__)

def _load_fs_cfg():
    try:
        from config_loader import cfg
        return {
            "store_path": getattr(cfg.feature_store, "store_path", "feature_store/store.db"),
            "cache_ttl" : int(getattr(cfg.feature_store, "cache_ttl_hours", 1)) * 3600,
            "version"   : getattr(cfg.feature_store, "version", "v1"),
            "staleness" : int(getattr(cfg.feature_store, "staleness_threshold_hours", 24)),
        }
    except Exception:
        return {"store_path": "feature_store/store.db", "cache_ttl": 3600, "version": "v1", "staleness": 24}

_FS_CFG = _load_fs_cfg()

DEFAULT_STORE_PATH  = os.getenv("FEATURE_STORE_PATH", _FS_CFG["store_path"])
DEFAULT_CACHE_TTL   = int(os.getenv("FEATURE_STORE_CACHE_TTL", str(_FS_CFG["cache_ttl"])))
DEFAULT_FS_VERSION  = _FS_CFG["version"]
DEFAULT_STALENESS_H = _FS_CFG["staleness"]

@dataclass
class FeatureGroup:
    """Définition d'un groupe de features (ex: 'property_base')."""
    name       : str
    description: str
    entity_key : str            # colonne clé (ex: "id")
    features   : list[str]      # liste des colonnes dans ce groupe
    ttl_hours  : int = 24       # durée de vie en cache
    version    : str = "v1"

    def to_dict(self) -> dict:
        return {
            "name"       : self.name,
            "description": self.description,
            "entity_key" : self.entity_key,
            "features"   : self.features,
            "ttl_hours"  : self.ttl_hours,
            "version"    : self.version,
        }

@dataclass
class FeatureStats:
    """Statistiques d'un groupe de features (pour monitoring)."""
    group_name  : str
    n_rows      : int
    n_features  : int
    written_at  : str
    version     : str
    checksum    : str           # hash du DataFrame: détecter les changements
    size_bytes  : int

class FeatureStore:
    """
    Feature Store SQLite-backed pour le pipeline Avito Real Estate.

    Supporte :
      - write()          : écriture d'un DataFrame de features
      - read()           : lecture par entity IDs
      - read_all()       : lecture complète d'un groupe
      - read_as_of()     : point-in-time correct (anti-leakage)
      - get_stats()      : statistiques du store
      - list_groups()    : liste des feature groups disponibles
      - get_training_dataset() : assemble tous les groups en un seul DataFrame
    """

    def __init__(
        self,
        store_path: str = DEFAULT_STORE_PATH,
        cache_ttl_s: int = DEFAULT_CACHE_TTL,
    ):
        self.store_path = store_path
        self._cache_ttl = cache_ttl_s
        self._memory_cache: dict[str, tuple[pd.DataFrame, float]] = {}

        Path(store_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        logger.info(f" FeatureStore initialisé → {store_path}")

    def _init_db(self) -> None:
        """Crée les tables SQLite si elles n'existent pas."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS feature_groups (
                    name        TEXT PRIMARY KEY,
                    description TEXT,
                    entity_key  TEXT NOT NULL,
                    features    TEXT NOT NULL,   -- JSON list
                    ttl_hours   INTEGER DEFAULT 24,
                    version     TEXT DEFAULT 'v1',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feature_values (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_name  TEXT NOT NULL,
                    entity_id   TEXT NOT NULL,
                    version     TEXT NOT NULL,
                    features    TEXT NOT NULL,   -- JSON dict {feature: value}
                    written_at  TEXT NOT NULL,
                    checksum    TEXT NOT NULL,
                    UNIQUE(group_name, entity_id, version)
                );

                CREATE INDEX IF NOT EXISTS idx_fv_group_entity
                    ON feature_values(group_name, entity_id);

                CREATE INDEX IF NOT EXISTS idx_fv_written_at
                    ON feature_values(group_name, written_at);

                CREATE TABLE IF NOT EXISTS feature_stats (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_name  TEXT NOT NULL,
                    n_rows      INTEGER,
                    n_features  INTEGER,
                    written_at  TEXT NOT NULL,
                    version     TEXT NOT NULL,
                    checksum    TEXT NOT NULL,
                    size_bytes  INTEGER
                );
            """)

    @contextmanager
    def _conn(self):
        """Connexion SQLite avec gestion auto-commit."""
        conn = sqlite3.connect(self.store_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def write(
        self,
        df: pd.DataFrame,
        group: "str | FeatureGroup",
        entity_key: str = "id",
        version: str = "v1",
        description: str = "",
        overwrite: bool = True,
    ) -> FeatureStats:
        """
        Écrit un DataFrame de features dans le Feature Store.

        Parameters
        ----------
        df         : DataFrame avec les features + colonne entity_key
        group      : nom du groupe ou objet FeatureGroup
        entity_key : colonne identifiant chaque entité (ex: "id")
        version    : version des features (ex: "v1", "v2")
        description: description du groupe
        overwrite  : remplacer les features existantes

        Returns
        -------
        FeatureStats avec les informations d'écriture
        """
        group_name = group.name if isinstance(group, FeatureGroup) else group

        if entity_key not in df.columns:
            df = df.copy()
            df[entity_key] = [f"row_{i}" for i in range(len(df))]

        feature_cols = [c for c in df.columns if c != entity_key]
        checksum     = self._checksum(df[feature_cols])
        written_at   = datetime.now().isoformat()

        logger.info(
            f"    FeatureStore.write → '{group_name}' v{version} | "
            f"{len(df)} entités | {len(feature_cols)} features"
        )

        t0 = time.perf_counter()

        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO feature_groups
                    (name, description, entity_key, features, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, COALESCE(
                    (SELECT created_at FROM feature_groups WHERE name=?), ?
                ), ?)
            """, (
                group_name,
                description or f"Feature group '{group_name}'",
                entity_key,
                json.dumps(feature_cols),
                version,
                group_name, written_at,
                written_at,
            ))

            if overwrite:
                conn.execute(
                    "DELETE FROM feature_values WHERE group_name=? AND version=?",
                    (group_name, version),
                )

            rows = []
            for _, row in df.iterrows():
                entity_id = str(row[entity_key])
                feat_dict = {
                    c: self._serialize_value(row[c])
                    for c in feature_cols
                }
                row_checksum = hashlib.md5(
                    json.dumps(feat_dict, sort_keys=True).encode()
                ).hexdigest()[:8]
                rows.append((
                    group_name, entity_id, version,
                    json.dumps(feat_dict), written_at, row_checksum,
                ))

            conn.executemany("""
                INSERT OR REPLACE INTO feature_values
                    (group_name, entity_id, version, features, written_at, checksum)
                VALUES (?, ?, ?, ?, ?, ?)
            """, rows)

            size_bytes = sum(len(r[3]) for r in rows)
            conn.execute("""
                INSERT INTO feature_stats
                    (group_name, n_rows, n_features, written_at, version, checksum, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (group_name, len(df), len(feature_cols), written_at, version, checksum, size_bytes))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        self._invalidate_cache(group_name)

        stats = FeatureStats(
            group_name = group_name,
            n_rows     = len(df),
            n_features = len(feature_cols),
            written_at = written_at,
            version    = version,
            checksum   = checksum,
            size_bytes = size_bytes,
        )

        logger.info(
            f"   PASS Écrit en {elapsed_ms:.1f}ms | "
            f"{size_bytes/1024:.1f} KB | checksum={checksum[:8]}"
        )
        return stats

    def read(
        self,
        group: str,
        entity_ids: Optional[list[str]] = None,
        version: str = "v1",
        features: Optional[list[str]] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Lit les features d'un groupe depuis le Feature Store.

        Parameters
        ----------
        group      : nom du groupe de features
        entity_ids : liste d'IDs à lire (None = tous)
        version    : version des features
        features   : sous-ensemble de features à retourner (None = toutes)
        use_cache  : utiliser le cache mémoire (TTL configuré)

        Returns
        -------
        DataFrame avec les features demandées
        """
        cache_key = f"{group}_{version}_{hash(str(entity_ids))}"

        if use_cache and cache_key in self._memory_cache:
            df_cached, ts = self._memory_cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                logger.debug(f"    Cache HIT → '{group}' v{version}")
                return df_cached[features] if features else df_cached

        t0 = time.perf_counter()

        with self._conn() as conn:
            if entity_ids is not None:
                placeholders = ",".join("?" * len(entity_ids))
                query = f"""
                    SELECT entity_id, features FROM feature_values
                    WHERE group_name=? AND version=? AND entity_id IN ({placeholders})
                """
                params = [group, version] + list(map(str, entity_ids))
            else:
                query = """
                    SELECT entity_id, features FROM feature_values
                    WHERE group_name=? AND version=?
                """
                params = [group, version]

            rows = conn.execute(query, params).fetchall()

        if not rows:
            logger.warning(
                f"   WARNING  FeatureStore.read → '{group}' v{version} : "
                f"aucune donnée trouvée"
            )
            return pd.DataFrame()

        records = []
        for row in rows:
            feat_dict = json.loads(row["features"])
            feat_dict["entity_id"] = row["entity_id"]
            records.append(feat_dict)

        df = pd.DataFrame(records)

        for col in df.columns:
            if col != "entity_id":
                try:
                    df[col] = pd.to_numeric(df[col])
                except (TypeError, ValueError):
                    pass

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"    FeatureStore.read → '{group}' v{version} | "
            f"{len(df)} entités | {elapsed_ms:.1f}ms"
        )

        if use_cache:
            self._memory_cache[cache_key] = (df, time.time())

        return df[features].copy() if features else df.copy()

    def read_all(self, group: str, version: str = "v1") -> pd.DataFrame:
        """Lit toutes les entités d'un groupe."""
        return self.read(group=group, version=version, use_cache=False)

    def read_as_of(
        self,
        group: str,
        timestamp: str,
        version: str = "v1",
    ) -> pd.DataFrame:
        """
        Point-in-time correct lookup: évite le data leakage.

        Retourne uniquement les features qui étaient disponibles
        AVANT le timestamp donné.

        Utile pour : backtesting, validation temporelle.

        Parameters
        ----------
        group     : nom du groupe
        timestamp : ISO datetime: ne retourner que les features écrites avant
        version   : version des features
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT entity_id, features FROM feature_values
                WHERE group_name=? AND version=? AND written_at <= ?
                ORDER BY written_at DESC
            """, (group, version, timestamp)).fetchall()

        if not rows:
            logger.warning(f"   WARNING  read_as_of → aucune donnée avant {timestamp}")
            return pd.DataFrame()

        records = []
        seen_entities = set()
        for row in rows:
            eid = row["entity_id"]
            if eid not in seen_entities:   # garder la version la plus récente avant timestamp
                seen_entities.add(eid)
                feat_dict = json.loads(row["features"])
                feat_dict["entity_id"] = eid
                records.append(feat_dict)

        df = pd.DataFrame(records)
        for col in df.columns:
            if col != "entity_id":
                try:
                    df[col] = pd.to_numeric(df[col])
                except (TypeError, ValueError):
                    pass

        logger.info(
            f"    read_as_of('{group}', {timestamp[:10]}) → {len(df)} entités"
        )
        return df

    def get_training_dataset(
        self,
        groups: list[str],
        version: str = "v1",
        join_key: str = "entity_id",
    ) -> pd.DataFrame:
        """
        Assemble plusieurs feature groups en un seul DataFrame de training.

        C'est le cœur du Feature Store : au lieu d'appeler extract + features
        à chaque fois, on lit directement les features pré-calculées.

        Parameters
        ----------
        groups   : liste de feature groups à joindre
        version  : version des features
        join_key : colonne de jointure

        Returns
        -------
        DataFrame avec toutes les features des groupes demandés
        """
        logger.info(f"    Assemblage dataset : {groups}")

        dfs = []
        for group in groups:
            df = self.read_all(group=group, version=version)
            if df.empty:
                logger.warning(f"   WARNING  Groupe '{group}' vide: ignoré")
                continue
            dfs.append(df)

        if not dfs:
            logger.error("   FAIL Aucun groupe disponible pour assembler le dataset")
            return pd.DataFrame()

        result = dfs[0]
        for df in dfs[1:]:
            cols_to_add = [c for c in df.columns if c not in result.columns or c == join_key]
            result = result.merge(df[cols_to_add], on=join_key, how="left")

        logger.info(
            f"   PASS Dataset assemblé : {len(result)} entités × {len(result.columns)} features"
        )
        return result

    def list_groups(self) -> list[dict]:
        """Liste tous les feature groups disponibles."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT fg.*, COUNT(fv.id) as n_entities
                FROM feature_groups fg
                LEFT JOIN feature_values fv ON fg.name = fv.group_name
                GROUP BY fg.name
            """).fetchall()

        return [
            {
                "name"       : row["name"],
                "description": row["description"],
                "entity_key" : row["entity_key"],
                "features"   : json.loads(row["features"]),
                "version"    : row["version"],
                "n_entities" : row["n_entities"],
                "updated_at" : row["updated_at"],
            }
            for row in rows
        ]

    def get_stats(self, group: Optional[str] = None) -> list[dict]:
        """Retourne les statistiques du Feature Store."""
        with self._conn() as conn:
            if group:
                rows = conn.execute(
                    "SELECT * FROM feature_stats WHERE group_name=? ORDER BY written_at DESC LIMIT 10",
                    (group,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM feature_stats ORDER BY written_at DESC LIMIT 50"
                ).fetchall()

        return [dict(row) for row in rows]

    def get_feature_freshness(self) -> dict[str, str]:
        """
        Retourne la date de dernière écriture pour chaque groupe.
        Utile pour détecter les features obsolètes.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT group_name, MAX(written_at) as last_written
                FROM feature_values
                GROUP BY group_name
            """).fetchall()

        freshness = {row["group_name"]: row["last_written"] for row in rows}

        now = datetime.now()
        for group, ts in freshness.items():
            last = datetime.fromisoformat(ts)
            age_h = (now - last).total_seconds() / 3600
            if age_h > 24:
                logger.warning(
                    f"   WARNING  Groupe '{group}' obsolète: "
                    f"dernière MAJ il y a {age_h:.1f}h"
                )

        return freshness

    def print_summary(self) -> None:
        """Affiche un résumé du Feature Store dans les logs."""
        groups = self.list_groups()
        freshness = self.get_feature_freshness()

        logger.info(f"\n{'='*60}")
        logger.info(f"    Feature Store Summary")
        logger.info(f"   Fichier : {self.store_path}")
        logger.info(f"{'='*60}")

        if not groups:
            logger.info("   Aucun feature group enregistré.")
        else:
            for g in groups:
                last = freshness.get(g["name"], "jamais")[:19]
                age_marker = ""
                if last != "jamais":
                    age_h = (datetime.now() - datetime.fromisoformat(last)).total_seconds() / 3600
                    age_marker = f"WARNING  ({age_h:.1f}h)" if age_h > 24 else "PASS"
                logger.info(
                    f"    {g['name']:<25} | v{g['version']} | "
                    f"{g['n_entities']:>6} entités | "
                    f"{len(g['features']):>3} features | "
                    f"{last} {age_marker}"
                )

        logger.info(f"{'='*60}\n")

    @staticmethod
    def _serialize_value(val: Any) -> Any:
        """Convertit les types numpy/pandas en types Python natifs pour JSON."""
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, (np.bool_,)):
            return bool(val)
        if isinstance(val, float) and np.isnan(val):
            return None
        return val

    @staticmethod
    def _checksum(df: pd.DataFrame) -> str:
        """Hash MD5 d'un DataFrame pour détecter les changements."""
        try:
            data_bytes = pd.util.hash_pandas_object(df).values.tobytes()
            return hashlib.md5(data_bytes).hexdigest()[:16]
        except Exception:
            return hashlib.md5(str(df.shape).encode()).hexdigest()[:16]

    def _invalidate_cache(self, group: str) -> None:
        """Invalide le cache mémoire pour un groupe."""
        keys_to_del = [k for k in self._memory_cache if k.startswith(f"{group}_")]
        for k in keys_to_del:
            del self._memory_cache[k]

def pipeline_write_features(
    df_train: pd.DataFrame,
    feature_names: list[str],
    store_path: str = DEFAULT_STORE_PATH,
    version: str = "v1",
) -> dict[str, FeatureStats]:
    """
    Écrit tous les feature groups après un training pipeline.

    Appelé depuis pipeline.py après le feature engineering.
    Organise les features en groupes logiques.

    Returns dict {group_name: FeatureStats}
    """
    fs = FeatureStore(store_path=store_path)

    groups: dict[str, list[str]] = {
        "property_base"   : [],
        "property_derived": [],
        "geographic"      : [],
        "temporal"        : [],
    }

    base_patterns      = ["surface_m2", "nb_chambres", "nb_salles_bain", "etage", "age_bien", "type_bien"]
    derived_patterns   = ["prix_par_m2", "log_", "score_luxe", "surface_x", "surface_par", "ratio_"]
    geographic_patterns= ["ville", "region", "median_", "rank_", "prix_median", "geo_"]
    temporal_patterns  = ["mois", "annee", "saison", "jour", "trimestre", "anciennete"]

    def _matches(col: str, patterns: list[str]) -> bool:
        return any(p in col.lower() for p in patterns)

    for col in feature_names:
        if _matches(col, base_patterns):
            groups["property_base"].append(col)
        elif _matches(col, geographic_patterns):
            groups["geographic"].append(col)
        elif _matches(col, temporal_patterns):
            groups["temporal"].append(col)
        else:
            groups["property_derived"].append(col)

    stats: dict[str, FeatureStats] = {}
    df_work = df_train.copy()

    df_work["entity_id"] = [f"train_{i}" for i in range(len(df_work))]

    for group_name, cols in groups.items():
        cols_available = [c for c in cols if c in df_work.columns]
        if not cols_available:
            logger.debug(f"   FeatureStore : groupe '{group_name}' vide: ignoré")
            continue

        group_df = df_work[["entity_id"] + cols_available].copy()
        group_descriptions = {
            "property_base"   : "Features brutes du bien immobilier",
            "property_derived": "Features calculées (ratios, interactions, scores)",
            "geographic"      : "Features géographiques (ville, région, statistiques locales)",
            "temporal"        : "Features temporelles (mois, saison, ancienneté)",
        }

        stat = fs.write(
            df          = group_df,
            group       = group_name,
            entity_key  = "entity_id",
            version     = version,
            description = group_descriptions.get(group_name, ""),
        )
        stats[group_name] = stat

    ref_path = Path(store_path).parent / "reference_data.pkl"
    df_ref = df_train[feature_names].copy() if all(c in df_train.columns for c in feature_names) else df_train.copy()
    with open(ref_path, "wb") as f:
        pickle.dump(df_ref, f)
    logger.info(f"    Référence drift sauvegardée → {ref_path}")

    fs.print_summary()
    return stats
