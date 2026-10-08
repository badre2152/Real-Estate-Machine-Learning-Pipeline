# Guide DVC

DVC est présent dans ce dépôt avec `dvc.yaml` et `.dvc/config`. Il sert à décrire les étapes et les artefacts du pipeline, mais le workflow DVC complet n'est **pas encore confirmé comme exécutable**.

## État du pipeline

Le fichier `dvc.yaml` définit cinq étapes : `extract`, `validate`, `featurize`, `train` et `evaluate`.

| Étape | Rôle déclaré | Réserve |
| --- | --- | --- |
| `extract` | Extraction PostgreSQL vers `data/raw/obt.parquet` | `src/extract.py` accepte désormais `--output` et enregistre le fichier Parquet avec `pyarrow` |
| `validate` | Validation des données via `src/validate_dvc.py` et rapport JSON | Enregistre le rapport même si les contrôles obligatoires échouent, puis retourne un code d'échec |
| `featurize` | `src/featurize_dvc.py` prépare les jeux train et test et enregistre les statistiques géographiques | Sorties : `train_fe.parquet`, `test_fe.parquet`, `geo_stats.pkl` |
| `train` | Entraînement à partir des fichiers Parquet brut et préparés | Réutilise `train_fe.parquet` et `test_fe.parquet` sans refaire les transformations |
| `evaluate` | Lecture de `models/results.json` | Dépend des artefacts de training |

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
