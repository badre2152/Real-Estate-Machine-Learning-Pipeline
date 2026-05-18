"""conftest.py — Configure sys.path pour les imports de tests.

Ajoute le répertoire racine ET src/ au sys.path afin que :
  - les modules dans src/ (api, pipeline, drift_detector...) s'importent directement
  - les fixtures partagées entre tous les tests soient disponibles
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
_SRC  = _ROOT / "src"

# Root en premier (pour conftest, config/, etc.)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# src/ ensuite (pour tous les modules du projet)
if str(_SRC) not in sys.path:
    sys.path.insert(1, str(_SRC))
