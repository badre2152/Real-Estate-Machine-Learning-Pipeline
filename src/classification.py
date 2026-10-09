"""
classification.py
-----------------
Modèle de classification pour prédire la catégorie de prix (bas/moyen/élevé).

Améliorations v2 :
  - XGBoost ajouté
  - Encodage ordinal correct (bas < moyen < élevé)
  - Calibration isotonique des probabilités (optionnelle)
  - Rapport de déséquilibre automatique
  - Logging structuré
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from logger_setup import get_logger

try:
    from config_loader import cfg as _cfg
    _RS      = int(_cfg.pipeline.random_state)
    _CV      = int(_cfg.classification.cv_folds)
    _CLASSES = list(_cfg.classification.ordered_classes)
except Exception:
    _RS, _CV, _CLASSES = 42, 5, ["bas", "moyen", "élevé"]

logger = get_logger(__name__)

ORDERED_CLASSES = _CLASSES

def _try_xgboost():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        return None

def get_classification_models() -> dict:
    """Retourne les modèles candidats."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=_RS),
        "RandomForest"      : RandomForestClassifier(n_estimators=100, random_state=_RS, n_jobs=1),
        "GradientBoosting"  : GradientBoostingClassifier(n_estimators=100, random_state=_RS),
    }
    XGB = _try_xgboost()
    if XGB:
        models["XGBoost"] = XGB(
            n_estimators=100, random_state=_RS, n_jobs=1,
            verbosity=0, eval_metric="mlogloss",
        )
    return models

def encode_target(y_train, y_test):
    train = pd.Series(y_train).astype("string").str.strip().str.lower()
    test = pd.Series(y_test).astype("string").str.strip().str.lower()
    if train.isna().any() or test.isna().any() or train.eq("").any() or test.eq("").any():
        raise ValueError("Classification labels must be present")
    encoder = LabelEncoder()
    encoded_train = encoder.fit_transform(train)
    if len(encoder.classes_) < 2:
        raise ValueError("Classification requires at least two property types")
    unknown = set(test.unique()) - set(encoder.classes_)
    if unknown:
        raise ValueError("Unseen property types in evaluation: " + ", ".join(sorted(unknown)))
    encoded_test = encoder.transform(test)
    return encoded_train, encoded_test, encoder

def check_class_balance(y_enc, label_encoder) -> float:
    """Affiche la distribution des classes et retourne le ratio min/max."""
    unique, counts = np.unique(y_enc, return_counts=True)
    ratio = counts.min() / counts.max()
    logger.info("   Distribution des classes :")
    for idx, cnt in zip(unique, counts):
        pct = cnt / len(y_enc) * 100
        logger.info(f"     {label_encoder.classes_[idx]:<10s} : {cnt:>5d} ({pct:.1f}%)")
    if ratio < 0.5:
        logger.warning(
            f"   ⚠️  Déséquilibre détecté (ratio={ratio:.2f}): envisager SMOTE ou class_weight"
        )
    return ratio

def train_classification(
    X_train, y_train, use_calibration: bool = False
):
    """
    Entraîne plusieurs modèles de classification et retourne le meilleur.

    Args:
        X_train:         Features d'entraînement.
        y_train:         Target catégorielle (bas/moyen/élevé).
        use_calibration: Calibrer les probabilités du meilleur modèle.

    Returns:
        (best_model, best_name, label_encoder)
    """
    logger.info("\n" + "=" * 50)
    logger.info("🧠 MODÈLE DE CLASSIFICATION: Type de Bien")
    logger.info("=" * 50)

    y_enc, _, le = encode_target(y_train, y_train)
    check_class_balance(y_enc, le)

    models  = get_classification_models()
    results = {}
    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=_RS)

    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_enc, cv=skf, scoring="f1_weighted", n_jobs=-1
        )
        results[name] = scores.mean()
        logger.info(
            f"   {name:<25s} → F1 (weighted) : {scores.mean():.4f} (±{scores.std():.4f})"
        )

    best_name  = max(results, key=results.get)
    best_model = models[best_name]
    logger.info(f"\n🏆 Meilleur modèle : {best_name} (F1={results[best_name]:.4f})")

    if best_name == "XGBoost":
        try:
            from sklearn.model_selection import train_test_split as _tts
            X_fit, X_val, y_fit, y_val = _tts(
                X_train, y_enc, test_size=0.15, random_state=_RS, stratify=y_enc
            )
            best_model.set_params(
                n_estimators    = 500,      # max estimators: early stopping va couper
                early_stopping_rounds = 20, # arrêt si pas d'amélioration sur 20 rounds
            )
            best_model.fit(
                X_fit, y_fit,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            logger.info(
                f"   🎯 XGBoost early stopping : "
                f"{best_model.best_iteration} estimateurs retenus / 500"
            )
        except Exception as es_exc:
            logger.warning(f"   ⚠️  Early stopping ignoré : {es_exc}")
            best_model.fit(X_train, y_enc)
    else:
        best_model.fit(X_train, y_enc)

    if use_calibration and hasattr(best_model, "predict_proba"):
        logger.info("   🎯 Calibration isotonique des probabilités ...")
        if hasattr(best_model, "set_params") and hasattr(best_model, "early_stopping_rounds"):
            best_model.set_params(early_stopping_rounds=None)
        best_model = CalibratedClassifierCV(best_model, method="isotonic", cv=5)
        best_model.fit(X_train, y_enc)

    return best_model, best_name, le

def evaluate_classification(model, X_test, y_test, label_encoder):
    """Évalue le modèle de classification sur le test set."""
    labels = pd.Series(y_test).astype("string").str.strip().str.lower()
    y_te_enc = label_encoder.transform(labels)
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_te_enc, y_pred)
    precision = precision_score(y_te_enc, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_te_enc, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_te_enc, y_pred, average="weighted", zero_division=0)

    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(
                y_te_enc, y_proba, multi_class="ovr", average="weighted"
            )
        except Exception as exc:
            logger.warning(f"   ⚠️  ROC-AUC non calculable : {exc}")

    logger.info("\n📊 RÉSULTATS CLASSIFICATION (Test Set) :")
    logger.info(f"   Accuracy  : {accuracy:.4f}")
    logger.info(f"   Precision : {precision:.4f}")
    logger.info(f"   Recall    : {recall:.4f}")
    logger.info(f"   F1-Score  : {f1:.4f}")
    if roc_auc:
        logger.info(f"   ROC-AUC   : {roc_auc:.4f}")
    logger.info(
        "\n📋 Rapport détaillé :\n"
        + classification_report(y_te_enc, y_pred, labels=np.arange(len(label_encoder.classes_)), target_names=label_encoder.classes_, zero_division=0)
    )

    if f1 >= 0.85:
        logger.info("🟢 Excellent modèle !")
    elif f1 >= 0.70:
        logger.info("🟡 Bon modèle")
    elif f1 >= 0.55:
        logger.info("🟠 Modèle moyen: revoir features ou SMOTE")
    else:
        logger.info("🔴 Modèle faible: déséquilibre ou features insuffisantes")

    return {
        "Accuracy": accuracy, "Precision": precision,
        "Recall": recall, "F1": f1, "ROC-AUC": roc_auc,
    }

def get_feature_importance(model, feature_names: list, top_n: int = 15):
    """Importance des features pour la classification."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, "coef_"):
        imp = pd.Series(np.abs(model.coef_).mean(axis=0), index=feature_names)
    else:
        return None
    imp = imp.sort_values(ascending=False)
    logger.info(f"\n🔑 Top {top_n} features (classification) :")
    for feat, val in imp.head(top_n).items():
        bar = "█" * int(val * 40)
        logger.info(f"   {feat:<35s} {bar} {val:.4f}")
    return imp

def save_model(model, label_encoder, path: str = "models/classification_model.pkl") -> None:
    """Sauvegarde le modèle + encodeur dans un seul fichier."""
    with open(path, "wb") as f:
        pickle.dump({"model": model, "label_encoder": label_encoder}, f)
    logger.info(f"💾 Modèle classification sauvegardé → {path}")

def load_model(path: str = "models/classification_model.pkl"):
    """Charge le modèle et l'encodeur depuis disque."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    logger.info(f"📂 Modèle classification chargé depuis {path}")
    return payload["model"], payload["label_encoder"]