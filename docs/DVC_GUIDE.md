# Guide DVC

DVC est présent dans ce dépôt avec `dvc.yaml` et `.dvc/config`. Il sert à décrire les étapes et les artefacts du pipeline, mais le workflow DVC complet n'est **pas encore confirmé comme exécutable**.

## État du pipeline

Le fichier `dvc.yaml` définit cinq étapes : `extract`, `validate`, `featurize`, `train` et `evaluate`.

| Étape | Rôle déclaré | Réserve |
| --- | --- | --- |
| `extract` | Extraction PostgreSQL vers `data/raw/obt.parquet` | `src/extract.py` accepte désormais `--output` et enregistre le fichier Parquet avec `pyarrow` |
| `validate` | Validation des données via `src/validate_dvc.py` et rapport JSON | Écrit `reports/dvc/validation_report.json` et retourne un code d'échec si une validation obligatoire échoue |
| `featurize` | `src/featurize_dvc.py` prépare les jeux train et test et enregistre les statistiques géographiques | Sorties : `train_fe.parquet`, `test_fe.parquet`, `geo_stats.pkl` |
| `train` | Entraînement à partir des fichiers Parquet brut et préparés | Réutilise `train_fe.parquet` et `test_fe.parquet` sans refaire les transformations |
| `evaluate` | `src/evaluate_dvc.py` affiche les métriques présentes dans `models/results.json` | Dépend du rapport produit par `train`, sans redéclarer le même fichier comme métrique DVC |

Les commandes d'extraction et de validation sont maintenant alignées sur les scripts Python. La phase de feature engineering dispose maintenant d'une commande dédiée. Le workflow DVC complet n'a pas encore été exécuté.

## Commandes de consultation

Depuis la racine du dépôt, après installation des dépendances :

```bash
dvc dag
dvc status
dvc metrics show
```

Ces commandes décrivent le workflow prévu et nécessitent un environnement DVC correctement configuré. Leur bon fonctionnement n'a pas été vérifié dans cette revue.

## Reproduction

```bash
dvc repro
```

L'étape `extract` accepte maintenant `--output`, mais son exécution exige un PostgreSQL accessible et une installation de `pyarrow`. Les autres étapes peuvent présenter des incompatibilités et n'ont pas été exécutées.

Le dépôt contient déjà `.dvc/config`, donc ne pas lancer `dvc init` une seconde fois sans besoin précis. Aucun stockage distant DVC opérationnel n'est configuré dans la version examinée. `dvc push` et `dvc pull` nécessitent d'abord un remote valide.

Les métriques citées dans les anciens exemples de cette documentation étaient illustratives, pas des performances mesurées et vérifiées du dépôt.

## Réutilisation des features

Le stage `train` transmet `--input-parquet data/raw/obt.parquet`, `--train-features data/processed/train_fe.parquet` et `--test-features data/processed/test_fe.parquet` à `src/pipeline.py`. Le pipeline valide toujours le dataset brut, puis charge les features préparées au lieu de refaire le split et le feature engineering. `geo_stats.pkl` reste une sortie de la phase `featurize`, non chargée directement dans ce mode. Le workflow complet n'a pas été exécuté.

## Métriques DVC

Le fichier `models/results.json` est déclaré une seule fois dans `dvc.yaml`, sous `train.metrics`. `evaluate` le référence seulement comme dépendance et affiche les valeurs réellement présentes. Les champs de régression `R2` et `MAE` sont requis, tandis que `F1` et `Accuracy` sont lus seulement si un résultat de classification existe. Aucune valeur zéro fictive n'est affichée pour une métrique absente. Les commandes n'ont pas été exécutées dans cette revue.

## Dépendances et sorties DVC

La validation DVC écrit son rapport dans `reports/dvc/validation_report.json` pour éviter un conflit avec le rapport `reports/validation_report.json` généré séparément par le pipeline d'entraînement. `train` dépend explicitement du rapport DVC, des fichiers train et test préparés et du Parquet brut. DVC ne déclare comme sorties de modèles que `best_regression_model.pkl`, `preprocessor.pkl` et `feature_names.pkl` ; les fichiers de classification ou de prédiction d'intervalles peuvent être absents suivant la configuration. Le répertoire `docs/plots/` n'est pas déclaré comme sortie DVC, car il contient déjà un CSV suivi par Git. Ces fichiers restent des artefacts produits par le pipeline, sans gestion DVC spécifique.

Cette correction porte sur la cohérence des dépendances et des chemins, et ne valide pas l'exécution complète. Les autres dépendances du pipeline peuvent encore nécessiter des adaptations.

## Paramètres et variables externes

Le script d'extraction utilise maintenant `database.table` depuis `config/config.yaml`. Les stages suivent également les modifications de `src/config_loader.py` et des réglages de validation pertinents.

DVC suit les valeurs littérales du fichier YAML. Les variables d'environnement sont résolues séparément par Python à l'exécution. Une modification de `.env` seule ne déclenche donc pas automatiquement une nouvelle exécution DVC. Le workflow complet n'a pas été exécuté.

## Git et sorties DVC

Comparaison avec les fichiers actuellement suivis dans Git : aucune sortie déclarée dans `dvc.yaml` ne correspond directement à un fichier suivi. Les sorties `data/`, les modèles générés et `reports/dvc/` doivent rester des artefacts locaux ou gérés par DVC, pas des ajouts Git ordinaires.

Attention : les anciens exemples `reports/validation_report.json`, `reports/monitoring_report.json` et `reports/ml_report.html` sont encore suivis dans Git. `src/pipeline.py` peut écrire à ces chemins pendant l'entraînement : les règles `.gitignore` n'empêchent pas la modification d'un fichier déjà suivi. Ne pas présenter une exécution DVC comme propre de toute modification Git tant que ces sorties historiques n'ont pas été séparées des sorties runtime. Les rapports historiques n'ont pas été supprimés.

## Rapports historiques et runtime

La configuration par défaut `paths.reports_dir` pointe désormais vers `reports/runtime/` pour les nouveaux rapports produits par `src/pipeline.py`. Les exemples historiques suivis dans `reports/` restent inchangés pendant les exécutions avec cette configuration. Le stage `validate` de DVC écrit séparément dans `reports/dvc/`. Les deux répertoires de sortie sont ignorés par Git. Une configuration personnalisée des chemins peut modifier ce comportement.

## Independent DVC calibration partition

The `featurize` stage now reserves the quantile calibration subset before feature engineering, calculates geographic statistics from model training rows only, and saves `data/processed/calibration_fe.parquet` alongside the training and test files. The `train` stage consumes this dedicated file instead of splitting already engineered features. Preprocessing fits on training rows and transforms the calibration rows. DVC tracks the calibration file and the interval method parameter. With bootstrap intervals, the calibration file is an empty schema-compatible Parquet artifact and is not consumed. This change has not been executed or validated; regenerate DVC outputs and the trained models before deploying it.

## DVC stage dependency audit

The featurize stage depends on the validation report, and training tracks changes to preprocessing and feature engineering code. Training declares the regression estimator, metrics, and prediction interval artifact as reproducible outputs. The interval method, confidence level, and bootstrap count are tracked parameters. The interval builder now respects its explicitly supplied configuration instead of replacing the arguments with global defaults. A fresh DVC reproduction and model retraining are needed before deployment; no stages or tests were executed as part of this audit.
