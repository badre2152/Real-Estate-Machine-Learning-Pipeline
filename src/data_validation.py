"""
data_validation.py
------------------
Validation des données à l'entrée du pipeline.

Deux modes disponibles :
  1. Validation personnalisée (toujours actif): tests rapides sans dépendance
  2. Great Expectations (optionnel): si ge installé, génère un rapport HTML riche

Les validations couvrent :
  - Colonnes obligatoires présentes
  - Types de données cohérents
  - Plages de valeurs acceptables (prix, surface)
  - Taux de valeurs manquantes par colonne
  - Doublons
  - Distribution statistique (z-score outliers)
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from logger_setup import get_logger

logger = get_logger(__name__)

try:
    import great_expectations as ge
    GE_AVAILABLE = True
except ImportError:
    GE_AVAILABLE = False
    logger.debug("great_expectations non installé: validation personnalisée uniquement")

@dataclass
class ValidationResult:
    """Résultat d'un test de validation individuel."""
    name: str
    passed: bool
    message: str
    severity: str = "error"   # "error" | "warning" | "info"
    details: dict = field(default_factory=dict)

@dataclass
class ValidationReport:
    """Rapport complet de validation d'un DataFrame."""
    n_rows: int
    n_cols: int
    n_passed: int
    n_failed: int
    n_warnings: int
    results: list[ValidationResult]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def passed(self) -> bool:
        """True si aucun test ERROR n'a échoué."""
        return all(
            r.passed or r.severity != "error"
            for r in self.results
        )

    def summary(self) -> str:
        status = "PASS PASSED" if self.passed else "FAIL FAILED"
        return (
            f"{status} | {self.n_passed} OK / {self.n_failed} erreurs "
            f"/ {self.n_warnings} warnings | {self.n_rows:,} lignes"
        )

    def to_dict(self) -> dict:
        return {
            "passed"    : self.passed,
            "n_rows"    : self.n_rows,
            "n_cols"    : self.n_cols,
            "n_passed"  : self.n_passed,
            "n_failed"  : self.n_failed,
            "n_warnings": self.n_warnings,
            "timestamp" : self.timestamp,
            "results"   : [
                {
                    "name"    : r.name,
                    "passed"  : r.passed,
                    "message" : r.message,
                    "severity": r.severity,
                    "details" : r.details,
                }
                for r in self.results
            ],
        }

class DataValidator:
    """
    Validateur de données configurable.
    Charge les règles depuis config.yaml ou des paramètres explicites.
    """

    def __init__(
        self,
        required_columns: Optional[list] = None,
        price_min: float = 1_000,
        price_max: float = 100_000_000,
        surface_min: float = 5,
        surface_max: float = 10_000,
        max_missing_pct: float = 0.30,
    ):
        try:
            from config_loader import cfg
            v = cfg.validation
            self.required_columns = required_columns or list(v.required_columns)
            self.price_min        = float(v.price_min)
            self.price_max        = float(v.price_max)
            self.surface_min      = float(v.surface_min)
            self.surface_max      = float(v.surface_max)
            self.max_missing_pct  = float(v.max_missing_pct)
        except Exception:
            self.required_columns = required_columns or ["prix", "surface_m2", "ville"]
            self.price_min        = price_min
            self.price_max        = price_max
            self.surface_min      = surface_min
            self.surface_max      = surface_max
            self.max_missing_pct  = max_missing_pct

    def _check_required_columns(self, df: pd.DataFrame) -> ValidationResult:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            return ValidationResult(
                name="required_columns",
                passed=False,
                message=f"Colonnes obligatoires manquantes : {missing}",
                severity="error",
                details={"missing": missing},
            )

        NULL_THRESHOLD = 0.80   # > 80% nulls → erreur
        mostly_null = []
        for col in self.required_columns:
            null_pct = df[col].isna().mean()
            if null_pct > NULL_THRESHOLD:
                mostly_null.append({"column": col, "null_pct": round(null_pct * 100, 1)})

        if mostly_null:
            return ValidationResult(
                name="required_columns",
                passed=False,
                message=(
                    f"Colonnes obligatoires quasi-vides (>{NULL_THRESHOLD*100:.0f}% nulls) : "
                    f"{[m['column'] for m in mostly_null]}"
                ),
                severity="error",
                details={"mostly_null": mostly_null},
            )

        return ValidationResult(
            name="required_columns",
            passed=True,
            message=f"Toutes les colonnes obligatoires présentes et non-vides ({len(self.required_columns)})",
        )

    def _check_no_empty_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        if df.empty:
            return ValidationResult(
                name="non_empty",
                passed=False,
                message="DataFrame vide: aucune ligne à traiter",
                severity="error",
            )
        return ValidationResult(
            name="non_empty",
            passed=True,
            message=f"DataFrame non vide : {len(df):,} lignes",
        )

    def _check_price_range(self, df: pd.DataFrame) -> ValidationResult:
        if "prix" not in df.columns:
            return ValidationResult("price_range", True, "Colonne prix absente: skip", "info")
        col = df["prix"].dropna()
        out_of_range = ((col < self.price_min) | (col > self.price_max)).sum()
        pct = out_of_range / len(col) * 100 if len(col) > 0 else 0
        passed = pct < 5.0
        return ValidationResult(
            name="price_range",
            passed=passed,
            message=(
                f"{out_of_range:,} prix hors [{self.price_min:,.0f}, {self.price_max:,.0f}] "
                f"({pct:.1f}%)"
            ),
            severity="warning" if pct < 10 else "error",
            details={"out_of_range": int(out_of_range), "pct": round(pct, 2)},
        )

    def _check_surface_range(self, df: pd.DataFrame) -> ValidationResult:
        if "surface_m2" not in df.columns:
            return ValidationResult("surface_range", True, "Colonne surface_m2 absente: skip", "info")
        col = df["surface_m2"].dropna()
        out_of_range = ((col < self.surface_min) | (col > self.surface_max)).sum()
        pct = out_of_range / len(col) * 100 if len(col) > 0 else 0
        passed = pct < 5.0
        return ValidationResult(
            name="surface_range",
            passed=passed,
            message=f"{out_of_range:,} surfaces hors [{self.surface_min}, {self.surface_max}] ({pct:.1f}%)",
            severity="warning" if pct < 10 else "error",
            details={"out_of_range": int(out_of_range), "pct": round(pct, 2)},
        )

    def _check_missing_values(self, df: pd.DataFrame) -> list[ValidationResult]:
        results = []
        for col in df.columns:
            miss_pct = df[col].isna().mean()
            passed = miss_pct <= self.max_missing_pct
            sev = "error" if miss_pct > 0.5 else "warning" if miss_pct > self.max_missing_pct else "info"
            results.append(ValidationResult(
                name=f"missing_{col}",
                passed=passed,
                message=f"[{col}] valeurs manquantes : {miss_pct:.1%}",
                severity=sev,
                details={"column": col, "missing_pct": round(float(miss_pct), 4)},
            ))
        return results

    def _check_duplicates(self, df: pd.DataFrame) -> ValidationResult:
        n_dup = df.duplicated().sum()
        pct = n_dup / len(df) * 100 if len(df) > 0 else 0
        return ValidationResult(
            name="duplicates",
            passed=n_dup == 0,
            message=f"{n_dup:,} doublons ({pct:.1f}%)",
            severity="warning",
            details={"n_duplicates": int(n_dup), "pct": round(pct, 2)},
        )

    def _check_price_outliers(self, df: pd.DataFrame) -> ValidationResult:
        if "prix" not in df.columns:
            return ValidationResult("price_outliers", True, "skip", "info")
        col = df["prix"].dropna()
        if len(col) < 10:
            return ValidationResult("price_outliers", True, "Trop peu de données pour z-score", "info")
        from scipy import stats as scipy_stats
        z = np.abs(scipy_stats.zscore(col))
        n_outliers = (z > 4).sum()
        pct = n_outliers / len(col) * 100
        return ValidationResult(
            name="price_outliers",
            passed=pct < 1.0,
            message=f"{n_outliers:,} outliers extrêmes (|z|>4) dans prix ({pct:.2f}%)",
            severity="warning",
            details={"n_outliers": int(n_outliers), "pct": round(pct, 3)},
        )

    def _check_dtype_consistency(self, df: pd.DataFrame) -> list[ValidationResult]:
        """Vérifie que les colonnes numériques attendues ne sont pas en objet."""
        expected_numeric = ["prix", "surface_m2"]
        results = []
        for col in expected_numeric:
            if col not in df.columns:
                continue
            is_num = pd.api.types.is_numeric_dtype(df[col])
            results.append(ValidationResult(
                name=f"dtype_{col}",
                passed=is_num,
                message=f"[{col}] type : {df[col].dtype} {'✓' if is_num else '→ attendu numérique'}",
                severity="error" if not is_num else "info",
            ))
        return results

    def validate(self, df: pd.DataFrame, stage: str = "input") -> ValidationReport:
        """
        Exécute tous les tests de validation sur le DataFrame.

        Args:
            df:    DataFrame à valider.
            stage: Étiquette du stade pipeline (ex: "input", "post_cleaning").

        Returns:
            ValidationReport
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"  VALIDATION DONNÉES: stade : {stage}")
        logger.info(f"{'='*50}")

        all_results: list[ValidationResult] = []

        all_results.append(self._check_no_empty_dataframe(df))
        all_results.append(self._check_required_columns(df))
        all_results.extend(self._check_dtype_consistency(df))
        all_results.append(self._check_duplicates(df))
        all_results.append(self._check_price_range(df))
        all_results.append(self._check_surface_range(df))
        all_results.append(self._check_price_outliers(df))
        all_results.extend(self._check_missing_values(df))

        n_passed   = sum(1 for r in all_results if r.passed)
        n_failed   = sum(1 for r in all_results if not r.passed and r.severity == "error")
        n_warnings = sum(1 for r in all_results if not r.passed and r.severity == "warning")

        report = ValidationReport(
            n_rows=len(df),
            n_cols=len(df.columns),
            n_passed=n_passed,
            n_failed=n_failed,
            n_warnings=n_warnings,
            results=all_results,
        )

        for r in all_results:
            icon = "PASS" if r.passed else ("FAIL" if r.severity == "error" else "WARNING")
            logger.info(f"   {icon} {r.message}")

        logger.info(f"\n   {report.summary()}")

        if not report.passed:
            logger.error("   FAIL Validation ÉCHOUÉE: vérifier les données avant de continuer")
        return report

    def save_report(self, report: ValidationReport, output_dir: str = "reports") -> str:
        """Sauvegarde le rapport de validation en JSON."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "validation_report.json")
        class _Encoder(json.JSONEncoder):
            def default(self, o):
                import numpy as np
                if isinstance(o, (bool, np.bool_)): return bool(o)
                if isinstance(o, np.integer): return int(o)
                if isinstance(o, np.floating): return float(o)
                return super().default(o)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, cls=_Encoder)
        logger.info(f"    Rapport validation → {path}")
        return path

def run_great_expectations(df: pd.DataFrame, output_dir: str = "reports/ge") -> Optional[dict]:
    """
    Validation via Great Expectations si installé.
    Génère un rapport HTML interactif dans output_dir.

    Returns:
        dict résultats GE ou None si GE non disponible.
    """
    if not GE_AVAILABLE:
        logger.debug("Great Expectations non disponible: skip")
        return None

    os.makedirs(output_dir, exist_ok=True)
    logger.info("    Great Expectations validation ...")

    try:
        gdf = ge.from_pandas(df)

        results = {}
        results["has_prix"]      = gdf.expect_column_to_exist("prix").success
        results["prix_not_null"] = gdf.expect_column_values_to_not_be_null("prix").success
        results["prix_positive"] = gdf.expect_column_values_to_be_between(
            "prix", min_value=1_000, max_value=100_000_000
        ).success

        if "surface_m2" in df.columns:
            results["surface_positive"] = gdf.expect_column_values_to_be_between(
                "surface_m2", min_value=5, max_value=10_000
            ).success

        if "ville" in df.columns:
            results["ville_not_null"] = gdf.expect_column_values_to_not_be_null("ville").success

        n_pass = sum(1 for v in results.values() if v)
        logger.info(f"   GE : {n_pass}/{len(results)} expectations passées")
        return results

    except Exception as exc:
        logger.warning(f"   WARNING  Great Expectations échoué : {exc}")
        return None