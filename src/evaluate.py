"""
evaluate.py
-----------
Évaluation et visualisation complète des modèles ML.

Améliorations v2 :
  - Analyse des erreurs par tranche de prix (Q1→Q5)
  - SHAP values pour l'interprétabilité (si shap installé)
  - Courbes ROC multiclasse (One-vs-Rest)
  - Courbe d'apprentissage (biais / variance)
  - Style visuel sombre professionnel
  - Logging structuré
"""

from logger_setup import get_logger
import os

import matplotlib
matplotlib.use("Agg")  # Backend non-interactif: compatible CI/CD
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

logger = get_logger(__name__)

try:
    from config_loader import cfg as _cfg
    PLOTS_DIR = str(_cfg.paths.plots_dir)
except Exception:
    PLOTS_DIR = os.getenv("PLOTS_DIR", "docs/plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor"  : "#1a1d2e",
    "axes.edgecolor"  : "#2d3150",
    "axes.labelcolor" : "#c0c8e0",
    "xtick.color"     : "#c0c8e0",
    "ytick.color"     : "#c0c8e0",
    "text.color"      : "#e8eaf6",
    "grid.color"      : "#2d3150",
    "grid.linestyle"  : "--",
    "grid.alpha"      : 0.5,
})
BLUE = "#5c8af7"
RED  = "#f75c8a"

def _save(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    plt.close()
    logger.info(f"   📊 → {path}")

def plot_prediction_vs_actual(y_test, y_pred, title="Régression : Prédit vs Réel"):
    """Scatter plot des valeurs prédites vs réelles. Bonne ligne = diagonale."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, y_pred, alpha=0.35, color=BLUE, edgecolors="none", s=25)
    lim = [min(np.min(y_test), np.min(y_pred)), max(np.max(y_test), np.max(y_pred))]
    ax.plot(lim, lim, "--", color=RED, linewidth=2, label="Prédiction parfaite")
    ax.set_xlabel("Prix Réel (MAD)"); ax.set_ylabel("Prix Prédit (MAD)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(True)
    _save(f"{PLOTS_DIR}/prediction_vs_actual.png")

def plot_residuals(y_test, y_pred):
    """Distribution des résidus. Bon modèle = résidus centrés en 0."""
    residuals = np.array(y_test) - np.array(y_pred)
    pct       = residuals / (np.abs(np.array(y_test)) + 1) * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(residuals, bins=45, color=BLUE, edgecolor="none", alpha=0.85)
    axes[0].axvline(0, color=RED, linewidth=2, linestyle="--")
    axes[0].set_title("Distribution des Résidus", fontweight="bold")
    axes[0].set_xlabel("Résidu (MAD)")

    axes[1].scatter(y_pred, residuals, alpha=0.3, color=BLUE, s=15, edgecolors="none")
    axes[1].axhline(0, color=RED, linewidth=2, linestyle="--")
    axes[1].set_title("Résidus vs Prédit", fontweight="bold")
    axes[1].set_xlabel("Prix Prédit"); axes[1].set_ylabel("Résidu")

    axes[2].hist(pct.clip(-100, 100), bins=45, color=RED, edgecolor="none", alpha=0.85)
    axes[2].axvline(0, color=BLUE, linewidth=2, linestyle="--")
    axes[2].set_title("Résidus Relatifs (%)", fontweight="bold")
    axes[2].set_xlabel("Erreur Relative (%)")

    _save(f"{PLOTS_DIR}/residuals.png")

def plot_error_by_price_range(y_test, y_pred):
    """
    MAE médiane par quintile de prix.
    Révèle si le modèle est moins précis sur les biens très chers.
    """
    df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
    df["abs_error"]   = np.abs(df["y_test"] - df["y_pred"])
    df["price_range"] = pd.qcut(df["y_test"], q=5, labels=["Q1","Q2","Q3","Q4","Q5"])

    fig, ax = plt.subplots(figsize=(9, 5))
    df.groupby("price_range", observed=True)["abs_error"].median().plot(
        kind="bar", ax=ax, color=BLUE, edgecolor="none", alpha=0.9
    )
    ax.set_title("Erreur Absolue Médiane par Tranche de Prix", fontweight="bold")
    ax.set_xlabel("Quintile (Q1=basse gamme, Q5=haute gamme)")
    ax.set_ylabel("MAE Médiane (MAD)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    _save(f"{PLOTS_DIR}/error_by_price_range.png")

def plot_feature_importance(model, feature_names, top_n=15, title="Importance des Features"):
    """Bar chart horizontal des features les plus importantes."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        imp = pd.Series(
            np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef),
            index=feature_names,
        )
    else:
        return

    top = imp.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
    bars = ax.barh(top.index, top.values, color=BLUE, edgecolor="none", alpha=0.9)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=8, color="#c0c8e0")
    slug = title.replace(" ", "_").lower()[:30]
    _save(f"{PLOTS_DIR}/feature_importance_{slug}.png")

def plot_learning_curve(model, X_train, y_train, scoring="r2", title="Courbe d'Apprentissage"):
    """Courbe d'apprentissage : détecte overfitting / underfitting."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 8), scoring=scoring,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_scores.mean(1), "o-", color=BLUE, label="Train", linewidth=2)
    ax.fill_between(train_sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.15, color=BLUE)
    ax.plot(train_sizes, val_scores.mean(1), "o-", color=RED, label="Validation", linewidth=2)
    ax.fill_between(train_sizes,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.15, color=RED)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Taille du Train Set"); ax.set_ylabel(scoring.upper())
    ax.legend(); ax.grid(True)
    _save(f"{PLOTS_DIR}/learning_curve.png")

def plot_confusion_matrix(y_test_enc, y_pred, class_names):
    """Matrice de confusion pour la classification."""
    cm = confusion_matrix(y_test_enc, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues"
    )
    ax.set_title("Matrice de Confusion", fontsize=14, fontweight="bold")
    _save(f"{PLOTS_DIR}/confusion_matrix.png")

def plot_roc_curves(model, X_test, y_test_enc, n_classes, class_names):
    """Courbes ROC multiclasse (One-vs-Rest)."""
    if not hasattr(model, "predict_proba"):
        return
    y_bin   = label_binarize(y_test_enc, classes=list(range(n_classes)))
    y_score = model.predict_proba(X_test)
    colors  = [BLUE, RED, "#f7c45c"]

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0,1],[0,1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Faux Positifs"); ax.set_ylabel("Vrais Positifs")
    ax.set_title("Courbes ROC Multiclasse (OvR)", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(True)
    _save(f"{PLOTS_DIR}/roc_curves.png")

def plot_shap_summary(model, X_test, feature_names, max_display=15):
    """SHAP values pour l'interprétabilité (pip install shap)."""
    try:
        import shap
        logger.info("   🔍 Calcul SHAP values ...")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:200])
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(sv, X_test[:200], feature_names=feature_names,
                          max_display=max_display, show=False)
        plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold")
        _save(f"{PLOTS_DIR}/shap_summary.png")
    except ImportError:
        logger.info("   ℹ️  shap non installé: ignoré (pip install shap)")
    except Exception as exc:
        logger.warning(f"   ⚠️  SHAP échoué : {exc}")

def run_full_evaluation(
    reg_model, clf_model, X_test,
    y_reg_test, y_clf_test, label_encoder, feature_names,
):
    """Lance toutes les visualisations d'évaluation en un seul appel."""
    logger.info("\n" + "=" * 50)
    logger.info("📊 ÉVALUATION COMPLÈTE DES MODÈLES")
    logger.info("=" * 50)

    y_reg_pred = reg_model.predict(X_test)

    logger.info("\n📈 Régression :")
    plot_prediction_vs_actual(np.array(y_reg_test), y_reg_pred)
    plot_residuals(np.array(y_reg_test), y_reg_pred)
    plot_error_by_price_range(np.array(y_reg_test), y_reg_pred)
    plot_feature_importance(reg_model, feature_names, title="Feature Importance: Régression")
    plot_learning_curve(reg_model, X_test, np.array(y_reg_test))
    plot_shap_summary(reg_model, X_test, feature_names)

    if clf_model is not None and y_clf_test is not None:
        logger.info("\n🧠 Classification :")
        from classification import ORDERED_CLASSES
        y_clf_enc = label_encoder.transform(
            pd.Series(y_clf_test).astype(str)
            .where(pd.Series(y_clf_test).astype(str).isin(ORDERED_CLASSES), "moyen")
        )
        y_pred_clf = clf_model.predict(X_test)
        plot_confusion_matrix(y_clf_enc, y_pred_clf, label_encoder.classes_)
        plot_roc_curves(clf_model, X_test, y_clf_enc, len(ORDERED_CLASSES), ORDERED_CLASSES)
        plot_feature_importance(clf_model, feature_names, title="Feature Importance: Classification")

    logger.info(f"\n✅ Tous les graphiques sauvegardés → {PLOTS_DIR}/")
