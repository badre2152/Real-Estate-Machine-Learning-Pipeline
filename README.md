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

Les endpoints API qui consultent les rapports de drift utilisent `REPORTS_DIR` s'il est défini, sinon `reports/runtime/`. Les composants Monitoring et Drift Detection utilisent également ce répertoire par défaut. Attention : dans Docker Compose, l'API et le pipeline doivent partager explicitement le même stockage de rapports pour que l'API puisse lire les rapports créés par le pipeline ; un chemin identique dans deux conteneurs isolés ne suffit pas. Aucun volume partagé n'a été ajouté pendant cette modification.

Dans Docker Compose, le volume nommé `reports_runtime` est désormais monté sur `/app/reports/runtime` dans l'API (lecture seule) et dans le conteneur `pipeline` (lecture et écriture). L'image initialise ce répertoire avec les permissions de l'utilisateur non privilégié. Le stockage est géré par Docker et n'apparaît pas automatiquement dans le répertoire `reports/runtime/` de la machine hôte. Le montage `./reports:/app/reports` du pipeline reste distinct et permet de conserver les autres rapports sur l'hôte. Cette configuration de volume n'a pas été exécutée ni vérifiée dans un conteneur.

## Modèles partagés avec Docker Compose

Le service `pipeline` écrit les modèles dans `/app/models` et le service `api` lit le même dossier via le montage hôte `./models`. Le montage API est en lecture seule. Le paramètre `MODELS_DIR` est défini explicitement sur les deux services et la configuration Python utilise la même variable pour enregistrer les modèles et le préprocesseur.

L'API charge les fichiers sérialisés au démarrage uniquement. Après un nouvel entraînement, redémarrer le service API pour qu'il recharge les modèles. Le succès de la lecture et de l'écriture dépend des permissions du dossier `./models` sur la machine hôte : préparer un dossier accessible à l'utilisateur non privilégié du conteneur. Ne jamais charger des fichiers pickle provenant de sources non fiables. Cette configuration n'a pas été exécutée durant la revue.

Au démarrage, FastAPI tente de charger chaque artefact pickle depuis `MODELS_DIR`. Un fichier manquant, illisible ou incompatible avec les classes Python installées génère un avertissement sans faire échouer tout le serveur. L'endpoint `/health` reste un contrôle de vie du processus ; `/ready` renvoie 503 si le modèle de régression ou le préprocesseur est absent. Les autres modèles sont facultatifs selon les fonctionnalités utilisées. Cette gestion n'a pas été vérifiée par une exécution du service. Charger uniquement des pickle produits par une source de confiance.

Les endpoints `/v1/predict` et `/v1/predict/batch` demandent désormais le modèle de régression et le préprocesseur. S'il manque l'un des deux, l'API retourne HTTP 503 avec un message générique et un request ID, au lieu de tenter une inférence sur des colonnes brutes. La version batch vérifie cette disponibilité avant de traiter les éléments. Les exceptions internes restent masquées au client. Aucun appel de prédiction n'a été exécuté pour vérifier le résultat.

Pour `/v1/predict/batch`, la liste accepte de 1 à 100 objets. Chaque propriété est validée indépendamment avec `PropertyInput` : une propriété mal formée reçoit une entrée d'erreur contenant son index (indexation à partir de zéro) et les noms des champs concernés, tandis que les propriétés valides continuent. L'ordre des entrées de réponse correspond à celui des propriétés soumises. Les détails bruts de validation et les données envoyées ne sont pas exposés dans ces erreurs. Une structure de requête incorrecte (par exemple une liste absente) reste rejetée globalement par la validation HTTP.

## Target leakage protection

The preprocessing stage excludes price-derived inputs and geographic price statistics from model features. The API no longer supplies placeholder values for those inputs. Existing saved models and preprocessors must be retrained together before deployment, as the input schema may differ. No training or runtime verification has been performed for this change.

## Classification target

The classification target is generated from `type_bien` only. Earlier fallbacks based on `piscine` or price bands could expose the label through model inputs or change the meaning of the target. If `type_bien` is absent, no classification target is synthesized. The geographic deviation feature that required the true price is no longer generated. The API field `price_category` currently refers to the classification prediction and may represent property type rather than a price band; its naming should be reviewed before a public API release. Existing training artifacts must be retrained before deployment. These changes have not been run or tested.

## Classification response semantics

Classification currently predicts the property type from `type_bien`. The internal training target remains named `categorie_prix` for compatibility with existing pipeline files, but it does not mean a price band. The prediction API now returns this value in `property_type`; the legacy `price_category` response field remains present with a null value to avoid falsely labelling a property type as a price band. Consumers should migrate to `property_type`. Retrain and redeploy model artifacts before relying on this output; older classifiers may represent a different target.

## Classification artifacts

Training saves a standalone classifier, a standalone label encoder, and a combined classification bundle. API startup can now load either standalone artifacts or extract the model and encoder from the combined bundle if needed. If a classifier is loaded without a label encoder, classification output is disabled rather than returning raw numeric class IDs as property types. Decode failures do not masquerade as valid property types. Saved artifact compatibility and runtime behavior have not been verified by executing the API.

## Regression artifacts and prediction validation

Training saves the regression estimator as both `regression_model.pkl` and `best_regression_model.pkl`. The API loads the preferred best-model file first, with the other as fallback. The prediction endpoint now rejects nonfinite or negative regression prices and inconsistent or nonfinite interval bounds; internal errors are not exposed to clients. The optional `pipeline.use_log_target` setting requires special attention: training can fit on log-transformed prices while the API currently interprets the raw estimator output as MAD. A consistent inverse transformation and artifact metadata still need to be implemented before enabling that setting in deployment. No training or runtime validation was performed.
