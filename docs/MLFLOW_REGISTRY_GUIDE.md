# Guide MLflow Registry

## Suivi et registre

Le pipeline utilise `MLflowTracker` pour enregistrer les expériences, métriques et artefacts. `MLflowRegistry` sert à enregistrer les modèles et gérer les étapes `Staging`, `Production` et `Archived`.

## Configuration locale avec Docker

```bash
cp .env.example .env
docker compose up -d
```

Configurer `DB_PASSWORD` et `API_KEYS` dans `.env` avant le démarrage.

L'interface MLflow est exposée localement sur `http://localhost:5000`. Entre conteneurs Docker, l'URI de suivi est `http://mlflow:5000`.

Pour un lancement sans Docker, `make mlflow-ui` utilise l'URI de suivi locale définie dans le Makefile. Il ne faut pas confondre cette commande avec le serveur MLflow démarré par Docker Compose.

## Enregistrement dans le pipeline

Après l'entraînement, `src/pipeline.py` journalise les modèles puis appelle `auto_register_and_promote` :

```python
result = auto_register_and_promote(
    run_id=tracker.run_id,
    model_name="avito-regression",
    artifact_path="regression_model",
    primary_metric="reg/R2",
    higher_is_better=True,
)
```

La classification utilise `avito-classification`, l'artefact `classification_model` et la métrique `clf/F1`.

La fonction enregistre une version puis compare les métriques avec la version en Production. Elle essaie ensuite de promouvoir les versions vers `Staging` ou `Production` selon la comparaison. Le succès effectif dépend de la disponibilité et de la configuration de MLflow ; il n'a pas été vérifié ici.

## Endpoints API

Les endpoints du registre exigent l'en-tête `X-API-Key` :

```text
GET /v1/registry
POST /v1/registry/promote?model_name=avito-regression&version=2&stage=Production
```

Exemples de commandes locales après configuration :

```bash
make registry-status API_KEY=your_actual_key
make registry-promote MODEL=avito-regression VERSION=2 STAGE=Production API_KEY=your_actual_key
```

Remplacer les valeurs d'exemple avant exécution. La commande de promotion modifie le registre des modèles : ne l'utiliser que pour une version volontairement sélectionnée.

## Limites

Les API de stages MLflow documentées ici correspondent à l'implémentation actuelle du projet. Elles ne constituent pas une validation du déploiement MLflow ni une garantie de compatibilité avec toutes les versions futures. Aucun modèle n'a été enregistré ou promu durant cette revue.
