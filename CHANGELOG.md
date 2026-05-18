# Changelog — ML Pipeline Avito Real Estate

Toutes les modifications notables sont documentées ici.
Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).

---

## [2.0.0] — 2026-05-14

### Ajouté
- **XGBoost** dans les modèles de régression et de classification
- **RandomizedSearchCV** pour l'optimisation des hyperparamètres (plus rapide que GridSearch)
- **MAPE** (Mean Absolute Percentage Error) comme métrique de régression complémentaire
- **SHAP values** pour l'interprétabilité des modèles tree-based (optionnel)
- **Courbe d'apprentissage** — détecte l'overfitting / underfitting
- **Analyse des erreurs par tranche de prix** (Q1→Q5)
- **Courbes ROC multiclasse** (One-vs-Rest)
- **Score de luxe composite** (piscine, ascenseur, garage, terrasse…)
- **Retry automatique** sur la connexion PostgreSQL (3 tentatives)
- **Validation du schéma OBT** à l'extraction
- **SMOTE automatique** si déséquilibre de classes détecté (ratio < 0.5)
- **Calibration isotonique** des probabilités de classification (optionnelle)
- **Timer par étape** dans le pipeline (durée précise de chaque phase)
- **Rapport JSON enrichi** (`models/results.json`) avec métriques + durées + options
- **Logging dual** : console + fichier horodaté dans `models/`
- **CLI complète** avec 6 options (`--optimize`, `--log-target`, `--smote`…)
- **Makefile** avec targets `test`, `lint`, `run`, `run-full`, `clean`
- **Tests améliorés** : 3 fichiers, ~60 tests, couverture ≥ 75%
- **GitHub Actions** : 3 jobs (`test`, `lint`, `pipeline-dry-run`)

### Modifié
- `prepare.py` : **OneHotEncoder** remplace LabelEncoder (évite les ordres artificiels)
- `prepare.py` : Détection **dynamique** des colonnes numériques/catégorielles
- `features.py` : Refactoring complet — chaque transformation dans sa propre fonction
- `evaluate.py` : Style visuel sombre professionnel sur tous les graphiques
- `pipeline.py` : Ajout du StepTimer et export JSON systématique
- `regression.py` : Support du **log-target** avec inverse-transform automatique

### Corrigé
- Division par zéro dans `prix_par_m2_calc` (surface = 0)
- Crash silencieux quand `categorie_prix` est absente de la table OBT
- Encodage ordinal incorrect (bas/moyen/élevé traités comme nominaux)
- `FutureWarning` pandas sur `is_categorical_dtype`

---

## [1.0.0] — 2025-09-01

### Ajouté
- Pipeline initial : extraction → préparation → régression → classification → évaluation
- Connexion PostgreSQL via SQLAlchemy
- Feature engineering de base (log_prix, prix_par_m2, surface_x_chambres)
- Modèles : RandomForest, GradientBoosting (régression + classification)
- Tests unitaires de base (test_extract, test_features, test_prepare)
- GitHub Actions CI basique
