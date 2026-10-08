"""
extract.py
----------
Connexion à PostgreSQL et extraction de la table OBT depuis ml_schema.

Améliorations v2 :
  - Retry automatique (3 tentatives) si la DB est indisponible
  - Validation du schéma extrait (colonnes obligatoires)
  - Rapport des valeurs manquantes au chargement
  - Support des filtres et du mode échantillon pour les tests
  - Logging structuré (remplace les print)
"""

import os
import time
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

from logger_setup import get_logger

load_dotenv()

logger = get_logger(__name__)

# Colonnes minimum attendues: adaptées à la table OBT réelle
REQUIRED_COLUMNS = ["prix", "surface_m2", "ville"]


def get_db_engine(max_retries: int = 3, retry_delay: int = 5):
    """
    Crée une connexion PostgreSQL avec retry automatique.
    Les paramètres viennent du fichier .env
    """
    host     = os.getenv("DB_HOST", "localhost")
    port     = os.getenv("DB_PORT", "5433")
    name     = os.getenv("DB_NAME", "real_estate_db")
    user     = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=name,
    )

    for attempt in range(1, max_retries + 1):
        engine = None
        try:
            engine = create_engine(url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"✅ Connexion PostgreSQL établie ({host}:{port}/{name})")
            return engine
        except OperationalError as exc:
            if engine is not None:
                engine.dispose()
            logger.warning("PostgreSQL connection attempt %s/%s failed (%s)", attempt, max_retries, type(exc).__name__)
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"❌ Impossible de se connecter à PostgreSQL après {max_retries} tentatives."
                ) from exc


def validate_schema(df: pd.DataFrame) -> None:
    """
    Vérifie que les colonnes essentielles sont présentes dans le DataFrame.
    Lève une ValueError si une colonne requise est manquante.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"❌ Colonnes manquantes dans la table OBT : {missing}\n"
            f"   Colonnes disponibles : {list(df.columns)}"
        )
    logger.info(f"✅ Schéma validé: {len(df.columns)} colonnes présentes")


# Allowlist des tables autorisées (anti SQL injection)
ALLOWED_TABLES = {
    "ml_schema.feature_store",
    "ml_schema.obt",
    "public.real_estate",
}

# Allowlist des colonnes de filtre autorisées (anti SQL injection)
ALLOWED_FILTER_COLS = {
    "ville", "type_bien", "region", "annee", "mois",
}


def _safe_table(table: str) -> str:
    """Valide le nom de table contre une allowlist: lève ValueError si non autorisé."""
    if table not in ALLOWED_TABLES:
        raise ValueError(
            f"❌ Table non autorisée : '{table}'. "
            f"Tables autorisées : {sorted(ALLOWED_TABLES)}"
        )
    return table


def _build_safe_query(
    table: str,
    filter_col: str | None,
    filter_val: str | None,
    limit: int | None,
) -> tuple[str, dict]:
    """
    Construit une requête SQL paramétrée: élimine le risque d'injection.

    Returns (query_string, params_dict) pour SQLAlchemy.
    """
    query = f"SELECT * FROM {_safe_table(table)}"
    params: dict = {}

    if filter_col is not None and filter_val is not None:
        if filter_col not in ALLOWED_FILTER_COLS:
            raise ValueError(
                f"❌ Colonne de filtre non autorisée : '{filter_col}'. "
                f"Colonnes autorisées : {sorted(ALLOWED_FILTER_COLS)}"
            )
        query += f" WHERE {filter_col} = :filter_val"
        params["filter_val"] = filter_val

    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"❌ limit doit être un entier positif, reçu : {limit!r}")
        query += " LIMIT :limit"
        params["limit"] = limit

    return query, params


def extract_obt(
    table: str = "ml_schema.feature_store",
    filter_col: str | None = None,
    filter_val: str | None = None,
    limit: int | None = None,
    # Rétro-compatibilité : ancien paramètre filters ignoré avec warning
    filters: str | None = None,
) -> pd.DataFrame:
    """
    Extrait la table OBT depuis ml_schema.feature_store.

    Args:
        table:      Nom complet de la table (doit être dans ALLOWED_TABLES).
        filter_col: Colonne de filtre (doit être dans ALLOWED_FILTER_COLS).
        filter_val: Valeur de filtre (passée comme paramètre SQL: safe).
        limit:      Nombre max de lignes (None = tout extraire).
        filters:    [DÉPRÉCIÉ] Ancien paramètre: ignoré, log un warning.

    Returns:
        DataFrame pandas nettoyé, prêt pour le feature engineering.
    """
    if filters is not None:
        logger.warning(
            "⚠️  Paramètre 'filters' déprécié (risque SQL injection): "
            "utiliser 'filter_col' + 'filter_val' à la place."
        )

    query, params = _build_safe_query(table, filter_col, filter_val, limit)
    engine = get_db_engine()

    logger.info(f"📥 Extraction depuis {table} ...")
    t0 = time.time()
    try:
        df = pd.read_sql(text(query), engine, params=params)
    finally:
        engine.dispose()
    elapsed = time.time() - t0

    logger.info(
        f"✅ {len(df):,} lignes extraites: {df.shape[1]} colonnes ({elapsed:.2f}s)"
    )

    # Validation du schéma
    validate_schema(df)

    # Rapport des valeurs manquantes
    missing_pct = df.isnull().mean() * 100
    top_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if not top_missing.empty:
        logger.info("📊 Valeurs manquantes détectées :")
        for col, pct in top_missing.items():
            logger.info(f"   {col:<35s} {pct:.1f}%")

    return df


def extract_sample(n: int = 1000) -> pd.DataFrame:
    """
    Extrait un échantillon aléatoire: utile pour les tests rapides.

    Applique la même validation de schéma que extract_obt() pour garantir
    que les tests utilisent des données structurellement identiques à la prod.

    Args:
        n: Nombre de lignes à extraire (défaut: 1000). Validé > 0.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"n doit être un entier positif, reçu : {n!r}")

    engine = get_db_engine()
    query = text("SELECT * FROM ml_schema.feature_store ORDER BY RANDOM() LIMIT :limit")
    logger.info(f"📥 Échantillon aléatoire ({n} lignes) ...")
    try:
        df = pd.read_sql(query, engine, params={"limit": n})
    finally:
        engine.dispose()

    # Même validation que extract_obt(): garantit la cohérence train/test
    validate_schema(df)

    logger.info(f"✅ Échantillon extrait et validé : {df.shape}")
    return df


if __name__ == "__main__":
    from logger_setup import get_logger as _get_logger
    _log = _get_logger(__name__)
    df = extract_obt()
    _log.info(f"\n{df.head().to_string()}")
    _log.info(f"\nDtypes :\n{df.dtypes.to_string()}")
    _log.info(f"\nShape : {df.shape}")
