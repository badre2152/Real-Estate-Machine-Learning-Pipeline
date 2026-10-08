"""
config_loader.py
----------------
Chargement centralisé de la configuration depuis config/config.yaml.

Usage :
    from config_loader import cfg
    model_dir = cfg.paths.models_dir
    test_size = cfg.pipeline.test_size
"""

import os
import re
import logging
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


class _AttrDict(dict):
    """Dict accessible via attributs : cfg.paths.models_dir"""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            return _AttrDict(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(f"Config key '{key}' introuvable")

    def __setattr__(self, key, value):
        self[key] = value


def _resolve_env(value: str) -> str:
    """
    Résout les références d'environnement du type ${VAR:default}.
    Ex : "${DB_HOST:localhost}" → valeur de DB_HOST ou "localhost".
    """
    pattern = r"\$\{(\w+)(?::([^}]*))?\}"

    def _replacer(match):
        var, default = match.group(1), match.group(2) or ""
        return os.getenv(var, default)

    return re.sub(pattern, _replacer, value)


def _walk_resolve(obj: Any) -> Any:
    """Parcourt récursivement le dict et résout les variables d'env."""
    if isinstance(obj, dict):
        return {k: _walk_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_resolve(i) for i in obj]
    if isinstance(obj, str):
        return _resolve_env(obj)
    return obj


def load_config(path: Path = _CONFIG_PATH) -> _AttrDict:
    """
    Charge le fichier YAML, résout les variables d'environnement
    et retourne un objet accessible par attributs.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config introuvable : {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    resolved = _walk_resolve(raw)
    logger.debug(f"Config chargée depuis {path}")
    return _AttrDict(resolved)


# Singleton global: importez `cfg` directement
cfg = load_config()
