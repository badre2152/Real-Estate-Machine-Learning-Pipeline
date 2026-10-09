"""
logger_setup.py
---------------
Logging professionnel avec rotation journalière, niveaux configurables,
et format structuré: remplace tous les print() du projet.

Usage :
    from logger_setup import get_logger
    logger = get_logger(__name__)
    logger.info("Démarrage du pipeline")
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

def get_logger(
    name: str,
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    backup_count: int = 7,
) -> logging.Logger:
    """
    Retourne un logger configuré avec :
      - Handler console (stdout): toujours actif
      - Handler fichier avec rotation journalière: si log_dir fourni

    Les paramètres sont lus depuis config.yaml si non spécifiés.

    Args:
        name:         Nom du module (__name__ recommandé).
        level:        Niveau de log (DEBUG/INFO/WARNING/ERROR).
        log_dir:      Dossier de sortie pour les fichiers de log.
        fmt:          Format du message de log.
        datefmt:      Format de la date.
        backup_count: Nombre de fichiers de log à conserver.

    Returns:
        Logger configuré.
    """
    try:
        from config_loader import cfg
        _level = level or cfg.logging.level
        _log_dir = log_dir or cfg.paths.logs_dir
        _fmt = fmt or cfg.logging.format
        _datefmt = datefmt or cfg.logging.datefmt
        _backup_count = backup_count or cfg.logging.backup_count
    except Exception:
        _level = level or "INFO"
        _log_dir = log_dir or "logs"
        _fmt = fmt or "%(asctime)s [%(levelname)s] %(name)s | %(message)s"
        _datefmt = datefmt or "%Y-%m-%d %H:%M:%S"
        _backup_count = backup_count

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    numeric_level = getattr(logging, _level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(fmt=_fmt, datefmt=_datefmt)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if _log_dir:
        log_path = Path(_log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            filename=log_path / "pipeline.log",
            when="midnight",
            backupCount=_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger

def configure_root_logger(level: Optional[str] = None, log_dir: Optional[str] = None) -> None:
    """
    Configure le logger racine une seule fois au démarrage du pipeline.
    Doit être appelé avant tout import de modules qui loggent.
    """
    try:
        from config_loader import cfg
        _level = level or cfg.logging.level
        _log_dir = log_dir or cfg.paths.logs_dir
    except Exception:
        _level = level or "INFO"
        _log_dir = log_dir or "logs"

    get_logger("root", level=_level, log_dir=_log_dir)

    for noisy in ("matplotlib", "PIL", "urllib3", "boto3", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
