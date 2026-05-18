# DVC — Versioning Données & Modèles

## Installation

```bash
pip install dvc
# Pour S3 :  pip install dvc-s3
# Pour GDrive : pip install dvc-gdrive
```

## Initialisation (première fois)

```bash
dvc init                        # initialise DVC dans le projet Git
dvc remote add -d myremote s3://my-bucket/avito-ml-dvc   # configurer remote
dvc remote modify myremote region eu-west-1
```

## Utilisation quotidienne

```bash
# Lancer le pipeline complet (reproduit uniquement les stages modifiés)
dvc repro

# Lancer un stage spécifique
dvc repro train

# Vérifier le statut du pipeline
dvc status

# Pousser les artefacts vers le remote
dvc push

# Récupérer les artefacts depuis le remote
dvc pull

# Comparer les métriques entre versions
dvc metrics diff HEAD~1

# Voir les métriques du run courant
dvc metrics show

# Afficher le graphe du pipeline
dvc dag
```

## Flux de travail typique

```bash
# 1. Modifier les paramètres dans config/config.yaml
# 2. Rejouer le pipeline
dvc repro

# 3. Comparer avec la version précédente
dvc metrics diff

# 4. Si les métriques s'améliorent, committer
git add dvc.lock models/results.json
git commit -m "feat: amélioration R² de 0.72 → 0.78"
dvc push
```

## Stages définis dans dvc.yaml

| Stage      | Input                  | Output                          |
|------------|------------------------|---------------------------------|
| extract    | DB PostgreSQL          | data/raw/obt.parquet            |
| validate   | obt.parquet            | reports/validation_report.json  |
| featurize  | obt.parquet            | data/processed/train_fe.parquet |
| train      | processed/             | models/*.pkl                    |
| evaluate   | models/results.json    | affichage métriques             |

## Métriques trackées

```bash
dvc metrics show models/results.json
```

Exemple de sortie :
```
Path                 regression.R2    regression.MAE    classification.F1
models/results.json  0.7842           43250             0.7614
```
