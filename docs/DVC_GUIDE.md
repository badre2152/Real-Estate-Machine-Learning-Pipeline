# Guide DVC

DVC est présent dans ce dépôt avec `dvc.yaml` et `.dvc/config`. Il sert à décrire les étapes et les artefacts du pipeline, mais le workflow DVC complet n'est **pas encore confirmé comme exécutable**.

## État du pipeline

Le fichier `dvc.yaml` définit cinq étapes : `extract`, `validate`, `featurize`, `train` et `evaluate`.

| Étape | Rôle déclaré | Réserve |
| --- | --- | --- |
| `extract` | Extraction PostgreSQL vers `data/raw/obt.parquet` | `src/extract.py` accepte désormais `--output` et enregistre le fichier Parquet avec `pyarrow` |
| `validate` | Validation des données via `src/validate_dvc.py` et rapport JSON | Enregistre le rapport même si les contrôles obligatoires échouent, puis retourne un code d'échec |
| `featurize` | Préparation des données et features | Dépend des fichiers Parquet et des signatures de fonctions |
| `train` | Exécution de `src/pipeline.py` | Le pipeline lit également PostgreSQL directement |
| `evaluate` | Lecture de `models/results.json` | Dépend des artefacts de training |

Les commandes d'extraction et de validation sont maintenant alignées sur les scripts Python. Les autres étapes DVC n'ont pas encore été vérifiées de bout en bout.

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
