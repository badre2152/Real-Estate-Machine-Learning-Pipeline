# 🏠 Avito Real Estate ML Pipeline

Pipeline Machine Learning complet pour la **prédiction et la classification de prix immobiliers au Maroc**.  
Connecté au projet ETL : [real-estate-pipeline](https://github.com/badre2152/real-estate-pipeline).

---

## Architecture du Pipeline

```
PostgreSQL
    ↓
ml_schema.feature_store
    ↓
extract.py
    ↓
features.py
    ↓
prepare.py
    ↓
regression.py + classification.py
    ↓
evaluate.py
    ↓
models/*.pkl + models/results.json
```

---

## Démarrage rapide

```bash
# 1. Cloner & installer
git clone https://github.com/badre2152/Real-Estate-Machine-Learning-Pipeline.git
cd Real-Estate-Machine-Learning-Pipeline
make install-dev

# 2. Configurer
cp .env.example .env
# Remplacer les valeurs change_me avant l'exécution

# 3. Lancer
make run             # Pipeline standard
make run-full        # Toutes options activées
make run-ci          # Mode CI (sans plots)
```

### Options CLI directes

```bash
python src/pipeline.py --help

# Exemples
python src/pipeline.py --log-target --smote
python src/pipeline.py --optimize --calibrate
python src/pipeline.py --no-plots --table ml_schema.feature_store
```

| Option | Effet |
|---|---|
| `--optimize` | RandomizedSearchCV (plus lent, meilleurs résultats) |
| `--log-target` | Régression sur log₁(prix), distribution plus gaussienne |
| `--smote` | SMOTE automatique si déséquilibre de classes |
| `--calibrate` | Calibration isotonique des probabilités |
| `--no-plots` | Désactiver les visualisations (CI/CD) |
| `--table` | Table OBT à extraire |

---

## Features créées (features.py)

| Feature | Description |
|---|---|
| `log_prix` | Log-transformation du prix (target régression) |
| `prix_zscore` + `is_prix_outlier` | Détection d'outliers |
| `prix_par_m2_calc` + `log_prix_par_m2` | Prix au m² |
| `surface_par_chambre` | Efficacité de surface par pièce |
| `surface_x_chambres` | Interaction surface × chambres |
| `ratio_chambres_bains` | Ratio pièces / salles de bain |
| `ville_prix_median` | Prix médian de la ville |
| `ecart_prix_ville` | Écart relatif vs médiane de la ville |
| `ville_rang_prix` | Rang de la ville par prix (1 = plus chère) |
| `region_prix_median` | Prix médian de la région |
| `score_luxe` | Score équipements haut de gamme (0–7) |
| `mois_annonce` | Mois de publication |
| `trimestre` | Trimestre |
| `est_weekend` | Annonce publiée le week-end |
| `jours_depuis_annonce` | Ancienneté de l'annonce |
| `categorie_prix` | Cible classification : bas / moyen / élevé |

---

## Modèles

### Régression: Prédiction du prix

Candidats : **Ridge**, **RandomForest**, **GradientBoosting**, **XGBoost**  
Métriques : MAE, MSE, RMSE, MAPE, R²

### Classification: Catégorie de prix

Candidats : **LogisticRegression**, **RandomForest**, **GradientBoosting**, **XGBoost**  
Métriques : Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## Visualisations générées

| Fichier | Description |
|---|---|
| `prediction_vs_actual.png` | Scatter prédit vs réel |
| `residuals.png` | Distribution des résidus (3 vues) |
| `error_by_price_range.png` | MAE par quintile de prix |
| `feature_importance_*.png` | Importance des features |
| `learning_curve.png` | Courbe d'apprentissage (biais/variance) |
| `shap_summary.png` | SHAP values (si `pip install shap`) |
| `confusion_matrix.png` | Matrice de confusion |
| `roc_curves.png` | Courbes ROC multiclasse |

---

## Tests

```bash
make test          # Tests rapides
make test-cov      # Tests + couverture HTML
make lint          # Linting ruff
```

**Seuil CI configuré : couverture minimale de 75%.** Cette valeur est un seuil de validation, pas une affirmation de couverture actuelle.

---

## Fichiers exportés

```
models/
├── regression_model.pkl       Meilleur modèle régression
├── classification_model.pkl   Meilleur modèle classification
├── preprocessor.pkl           Scaler + Encodeur (inférence future)
├── results.json               Métriques + durées + options
└── pipeline_YYYYMMDD.log      Log complet de l'exécution
```


## Auteur

**BRAHIM BADRE**  
Data Analyst | Data Engineer
