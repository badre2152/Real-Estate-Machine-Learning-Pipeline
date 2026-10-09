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
make install

# 2. Configurer
cp .env.example .env
# Remplacer les valeurs change_me avant l'exécution

# 3. Lancer
make run             # Pipeline standard
make run-full        # Toutes options activées
make run-no-plots    # Exécution sans visualisations
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
| `--no-plots` | Désactiver les visualisations |
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
| `score_luxe` | Score équipements haut de gamme (0 à 7) |
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

## Sécurité de l'API

Le proxy Nginx limite les requêtes par adresse IP. FastAPI applique une limite supplémentaire par clé API authentifiée. Le compteur FastAPI est conservé en mémoire par processus : avec plusieurs workers, il ne constitue pas un quota global partagé. Une limite centralisée nécessiterait un stockage commun tel que Redis.

Pour un déploiement avec `ENVIRONMENT=production`, remplacer `API_KEYS=change_me_api_key` par une clé forte et définir `CORS_ORIGINS` avec les origines HTTPS autorisées, séparées par des virgules. La valeur `*` est refusée en production. Ne pas versionner le fichier `.env`. Pour Docker Compose, définir également une valeur non vide pour `DB_PASSWORD` dans `.env`; le démarrage est désormais refusé si cette variable manque.

## Déploiement
**Artefacts conservés dans Git :** `feature_store/reference_data.pkl` est une référence générée consommée par l'endpoint de détection de drift. Sa structure et son contenu binaire n'ont pas été audités ; ne pas présumer qu'elle est anonymisée. Les rapports versionnés dans `reports/` sont des instantanés historiques et ne garantissent pas le succès de toutes les validations : l'exemple `validation_report.json` signale des contrôles en échec. Les nouvelles références `.pkl` et pages `.html` générées sont ignorées par Git ; les fichiers déjà suivis ne sont pas supprimés par `.gitignore`.

Le build Docker de l'API utilise `requirements-api.txt`, tandis que le service `pipeline` installe les dépendances complètes de `requirements.txt` via `REQUIREMENTS_FILE`. L'API conserve notamment MLflow, scikit-learn et XGBoost pour ses endpoints et ses modèles sérialisés. Les dépendances de SHAP, DVC et des rapports restent dans l'image d'entraînement. Ces images n'ont pas été construites ni exécutées dans cette revue.

**Render Blueprint :** `render.yaml` déclare uniquement l'API web. Une instance PostgreSQL existante est requise ; configurer `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER` et `DB_PASSWORD` dans le dashboard Render, ainsi que `API_KEYS` et `CORS_ORIGINS`. Le Blueprint ne crée plus de base de données ni de disque persistant. Vérifier l'offre Render et les paramètres de déploiement actuellement disponibles avant de déployer. Aucun plan payant n'est activé par ce changement. La configuration Docker Compose locale est indépendante.


Aucun workflow GitHub Actions de CI ou CD n'est configuré dans ce dépôt. Le build Docker et tout déploiement éventuel doivent être lancés et vérifiés manuellement.

Le proxy Nginx est optionnel et se lance avec `docker compose --profile nginx up -d`. Il attend que l'API soit healthy avant de démarrer. Pour activer HTTPS, fournir les certificats `nginx/certs/fullchain.pem` et `nginx/certs/privkey.pem` avant de lancer ce profil. Sans ces fichiers, Nginx ne peut pas démarrer avec la configuration SSL actuelle.

Les ports PostgreSQL (5433), FastAPI (8000) et MLflow (5000) sont liés à 127.0.0.1 sur la machine hôte. Ils restent accessibles localement, mais pas directement depuis le réseau externe par ces ports. Les conteneurs communiquent entre eux via le réseau Docker. Le profil Nginx publie les ports 80 et 443 et nécessite des certificats HTTPS valides pour démarrer.

---

## Modèles et fichiers générés

Le pipeline génère des artefacts de modèles dans `models/`, notamment `best_regression_model.pkl`, `best_classification_model.pkl` et `preprocessor.pkl`. Ces fichiers ne sont pas inclus dans les images Docker par défaut. Pour servir des prédictions, fournir des modèles compatibles dans `MODELS_DIR`.

Les fichiers de `reports/` actuellement versionnés sont des exemples historiques. Certains contrôles du rapport de validation ont échoué ; ils ne démontrent pas une validation complète des données.

## Documentation

- [Guide de déploiement](docs/DEPLOYMENT_GUIDE.md)
- [Guide DVC](docs/DVC_GUIDE.md)
- [Guide MLflow Registry](docs/MLFLOW_REGISTRY_GUIDE.md)
- [Contribution](CONTRIBUTING.md)

## Auteur

**BRAHIM BADRE**  
Data Analyst | Data Engineer

Les nouveaux rapports produits par le pipeline sont enregistrés dans `reports/runtime/`, via `paths.reports_dir` de `config/config.yaml`. Les exemples historiques déjà suivis dans `reports/` sont conservés et ne sont plus écrasés par le pipeline avec la configuration par défaut. Le répertoire runtime est ignoré par Git. Les sorties de validation DVC restent séparées dans `reports/dvc/`.
